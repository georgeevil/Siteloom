"""The web import wizard (CLD-27).

The wizard's reason to exist is that an operator should not need a shell
to register an archive. That is also precisely what makes it dangerous:
a text input that takes a filesystem path is an arbitrary-path read of
the host unless something confines it. Most of what is worth asserting
here is that confinement, and that the two phases stay honest about
which one is expensive and what `failed` means.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import LibraryItem, LibrarySource, get_session, init_db, make_engine
from siteloom.web.app import create_app
from siteloom.web.library_routes import ImportPathError, resolve_import_path


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "photos"
    (root / "trip").mkdir(parents=True)
    for n in range(3):
        (root / "trip" / f"p{n}.jpg").write_bytes(b"\xff\xd8\xff\xdb" + b"0" * 64)
    outside = tmp_path / "secrets"
    outside.mkdir()
    (outside / "id_rsa").write_text("nope")

    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.library.import_roots = [str(root)]
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    client = TestClient(create_app(config))
    client.sessionmaker = get_session(engine)
    return client, root, outside, config


# -- containment ---------------------------------------------------------


def test_a_path_outside_every_root_is_refused(env):
    _, root, outside, config = env
    with pytest.raises(ImportPathError):
        resolve_import_path(str(outside), config.library.import_roots)


def test_traversal_out_of_a_root_is_refused(env):
    """The string starts inside the root; the resolved path does not."""
    _, root, outside, config = env
    with pytest.raises(ImportPathError):
        resolve_import_path(f"{root}/../secrets", config.library.import_roots)


def test_a_sibling_directory_sharing_a_prefix_is_refused(env):
    """`/x/photos-backup` starts with `/x/photos` as a string. The gate is
    is_relative_to over resolved paths, never a prefix test (CLD-49)."""
    _, root, _, config = env
    sibling = root.parent / f"{root.name}-backup"
    sibling.mkdir()
    with pytest.raises(ImportPathError):
        resolve_import_path(str(sibling), config.library.import_roots)


def test_a_symlink_leaving_the_root_is_refused(env):
    """Resolving the candidate last is what catches this — a link living
    inside an allowed root is not itself permission to read its target."""
    _, root, outside, config = env
    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(ImportPathError):
        resolve_import_path(str(link), config.library.import_roots)


def test_the_root_itself_and_paths_under_it_are_allowed(env):
    _, root, _, config = env
    assert resolve_import_path(str(root), config.library.import_roots) == root.resolve()
    assert resolve_import_path(
        f"{root}/trip", config.library.import_roots
    ) == (root / "trip").resolve()


def test_no_configured_roots_means_no_web_import(env):
    """Empty is the default, and it disables the form rather than
    defaulting to the whole filesystem."""
    _, root, _, _ = env
    with pytest.raises(ImportPathError):
        resolve_import_path(str(root), [])


def test_a_relative_path_is_refused(env):
    _, _, _, config = env
    with pytest.raises(ImportPathError):
        resolve_import_path("trip", config.library.import_roots)


# -- the wizard ----------------------------------------------------------


def test_wizard_explains_itself_when_no_roots_are_configured(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/n.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    init_db(make_engine(config.storage.db_url))
    body = TestClient(create_app(config)).get("/library/import").text
    assert "import_roots" in body
    # It must not present a form that cannot work.
    assert 'name="path"' not in body


def test_refused_path_returns_the_form_with_a_reason(env):
    client, _, outside, _ = env
    r = client.post("/library/import/source", data={"path": str(outside)})
    assert r.status_code == 400
    assert "outside every configured import root" in r.text
    # And nothing was registered.
    with client.sessionmaker() as s:
        assert s.query(LibrarySource).count() == 0


def test_scan_registers_items_without_decoding_them(env):
    """Step 1 is cheap by design: files become pending rows, and the
    operator sees real counts before committing to the expensive pass."""
    client, root, _, _ = env
    r = client.post(
        "/library/import/source", data={"path": f"{root}/trip", "kind": "directory"}
    )
    assert r.status_code == 200, r.text[:400]
    assert "now pending" in r.text
    with client.sessionmaker() as s:
        source = s.query(LibrarySource).one()
        items = s.query(LibraryItem).filter_by(source_id=source.id).all()
        assert len(items) == 3
        assert {i.status for i in items} == {"pending"}


def test_scan_is_idempotent_over_the_same_directory(env):
    """Re-submitting the form must not duplicate the source or its rows."""
    client, root, _, _ = env
    for _ in range(2):
        client.post("/library/import/source", data={"path": f"{root}/trip"})
    with client.sessionmaker() as s:
        assert s.query(LibrarySource).count() == 1
        assert s.query(LibraryItem).count() == 3


def test_done_page_keeps_failed_out_of_pending(env):
    """Nothing re-queues a failed item, so a screen that folds the two
    together reports stalled work as queued."""
    client, root, _, _ = env
    client.post("/library/import/source", data={"path": f"{root}/trip"})
    with client.sessionmaker() as s:
        source = s.query(LibrarySource).one()
        item = s.query(LibraryItem).first()
        item.status = "failed"
        item.error = "cv2: bad header"
        s.commit()
        source_id = source.id

    body = client.get("/library/import/done", params={"source_id": source_id}).text
    assert "retry-failed" in body
    assert ">1<" in body  # the failed count stands on its own


def test_done_page_404s_for_an_unknown_source(env):
    client, _, _, _ = env
    assert client.get(
        "/library/import/done", params={"source_id": 999}
    ).status_code == 404
