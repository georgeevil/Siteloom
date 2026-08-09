"""The wizard's Google Takeout branch (CLD-92).

The bug this file exists for was invisible: choosing "Google Takeout" set
`LibrarySource.kind` and then ran an ordinary directory index. That
*succeeds*. Items get registered, the job completes, the done page shows
green numbers — and no sidecar is read, no `people` tag consulted, no
name proposed. Nothing distinguishes it from a working import except
asking which importer actually ran, so that is what most of these assert.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.library.takeout import preview_tree
from siteloom.store import (
    Annotation,
    LibraryItem,
    LibrarySource,
    OperationRun,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import library_routes
from siteloom.web.app import create_app

JPEG = b"\xff\xd8\xff\xdb" + b"0" * 64


def sidecar(path, title, people):
    path.write_text(json.dumps({"title": title, "people": [{"name": n} for n in people]}))


@pytest.fixture
def env(tmp_path):
    """A small Takeout-shaped tree and a plain one, under one root."""
    root = tmp_path / "archives"
    take = root / "Takeout" / "Google Photos" / "2019"
    take.mkdir(parents=True)
    for n in range(3):
        (take / f"IMG_{n}.jpg").write_bytes(JPEG)
        sidecar(take / f"IMG_{n}.jpg.supplemental-metadata.json", f"IMG_{n}.jpg", ["Ana"])
    # Google's edited copy: same moment as IMG_0, no sidecar of its own.
    (take / "IMG_0-edited.jpg").write_bytes(JPEG)

    plain = root / "plain"
    plain.mkdir()
    (plain / "a.jpg").write_bytes(JPEG)

    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/t.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.library.import_roots = [str(root)]
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    client = TestClient(create_app(config))
    client.sessionmaker = get_session(engine)
    return client, take, plain


class Spy:
    """Records that it was constructed and with what."""

    calls: list = []

    def __init__(self, indexer, auto_verify_unambiguous=True, progress=None):
        self.auto_verify = auto_verify_unambiguous
        Spy.calls.append(self)

    def import_tree(self, root, **kw):
        self.root = str(root)
        self.kw = kw


@pytest.fixture
def spy(monkeypatch):
    Spy.calls = []
    monkeypatch.setattr("siteloom.library.takeout.TakeoutImporter", Spy)
    processed = []
    monkeypatch.setattr(
        "siteloom.library.indexer.LibraryIndexer.process",
        lambda self, **kw: processed.append(kw),
    )
    return Spy.calls, processed


def start(client, path, source_kind, **form):
    """Walk the wizard end to end and wait for the background job.

    `form` goes to the index step only, so a test can post a `kind` there
    that contradicts the one step 1 saved.
    """
    client.post(
        "/library/import/source", data={"path": str(path), "kind": source_kind}
    )
    with client.sessionmaker() as s:
        source_id = s.query(LibrarySource).order_by(LibrarySource.id.desc()).first().id
    client.post(
        "/library/import/index",
        data={"source_id": source_id, **form},
        follow_redirects=False,
    )
    thread = library_routes._import_state["thread"]
    if thread is not None:
        thread.join(timeout=10)
    return source_id


# -- the gap itself ------------------------------------------------------


def test_a_takeout_source_runs_the_takeout_importer(env, spy):
    client, take, _ = env
    calls, processed = spy
    start(client, take, "takeout")
    assert len(calls) == 1, "TakeoutImporter never ran — this is the CLD-92 bug"
    assert calls[0].root == str(take)
    assert not processed, "ran a plain directory index over a Takeout tree"


def test_a_directory_source_still_runs_the_ordinary_indexer(env, spy):
    """The branch must not capture the path it was added beside."""
    client, _, plain = env
    calls, processed = spy
    start(client, plain, "directory")
    assert not calls
    assert len(processed) == 1


def test_the_kind_that_decides_is_the_stored_one_not_the_form(env, spy):
    """Step 2 posts only a source_id. Re-deciding from a resubmitted form
    field would let the expensive step disagree with what step 1 saved."""
    client, take, _ = env
    calls, processed = spy
    start(client, take, "takeout", kind="directory")
    assert len(calls) == 1
    assert not processed


# -- step 1 must not register what the importer will skip ----------------


def test_step_one_registers_nothing_for_a_takeout_tree(env):
    """`scan()` would register the `-edited` copies that import_tree
    skips, leaving them pending forever until some later `library index`
    picks them up and seeds the gallery with near-duplicates."""
    client, take, _ = env
    client.post("/library/import/source", data={"path": str(take), "kind": "takeout"})
    with client.sessionmaker() as s:
        assert s.query(LibraryItem).count() == 0
        assert s.query(LibrarySource).one().kind == "takeout"


def test_step_one_still_registers_items_for_a_plain_directory(env):
    client, _, plain = env
    client.post("/library/import/source", data={"path": str(plain), "kind": "directory"})
    with client.sessionmaker() as s:
        assert s.query(LibraryItem).count() == 1


def test_the_preview_counts_what_will_be_imported_not_what_is_there(env):
    _, take, _ = env
    preview = preview_tree(take)
    assert preview.media == 3          # the edited copy is not one of them
    assert preview.derivatives == 1
    assert preview.sidecars == 3
    assert preview.looks_like_takeout


def test_a_tree_with_no_sidecars_says_it_has_nothing_to_propose(env):
    client, _, plain = env
    preview = preview_tree(plain)
    assert not preview.looks_like_takeout
    body = client.post(
        "/library/import/source", data={"path": str(plain), "kind": "takeout"}
    ).text
    assert "nothing to propose" in body


# -- auto-verify ---------------------------------------------------------


def test_auto_verify_is_off_unless_asked_for(env, spy):
    """`verified` is what a training export reads as ground truth. A
    one-click web action must not author it on the operator's behalf."""
    client, take, _ = env
    calls, _ = spy
    start(client, take, "takeout")
    assert calls[0].auto_verify is False


def test_auto_verify_is_honoured_when_asked_for(env, spy):
    client, take, _ = env
    calls, _ = spy
    start(client, take, "takeout", auto_verify="1")
    assert calls[0].auto_verify is True


def test_the_resume_command_carries_the_auto_verify_choice(env, spy):
    """Dropping a flag on resume is the bug _resume_command was written
    for, and this is the flag whose loss writes unreviewed rows."""
    client, take, _ = env
    start(client, take, "takeout")
    with client.sessionmaker() as s:
        run = s.query(OperationRun).filter_by(kind="takeout-import").one()
        assert "--no-auto-verify" in run.resume_command
        assert str(take) in run.resume_command

    start(client, take, "takeout", auto_verify="1")
    with client.sessionmaker() as s:
        run = (
            s.query(OperationRun)
            .filter_by(kind="takeout-import")
            .order_by(OperationRun.id.desc())
            .first()
        )
        assert "--no-auto-verify" not in run.resume_command


def test_the_run_is_recorded_as_a_takeout_import_not_a_library_index(env, spy):
    """/jobs is where an operator looks for it, under the name of the
    thing they started."""
    client, take, _ = env
    start(client, take, "takeout")
    with client.sessionmaker() as s:
        assert s.query(OperationRun).filter_by(kind="library-index").count() == 0
        assert s.query(OperationRun).filter_by(kind="takeout-import").count() == 1


# -- the done page -------------------------------------------------------


def test_the_done_page_reports_faces_not_index_status(env, spy):
    """A finished import leaves every item `pending` — it attaches
    annotations and never marks an item indexed. Reporting the ordinary
    status counts would show a working run as "0 indexed, 3 pending"."""
    client, take, _ = env
    source_id = start(client, take, "takeout")
    with client.sessionmaker() as s:
        item = LibraryItem(
            source_id=source_id, path=str(take / "IMG_0.jpg"), kind="image",
            status="pending", mtime=datetime(2026, 8, 5, 12, 0),
        )
        s.add(item)
        s.flush()
        s.add(Annotation(
            item_id=item.id, bbox="[0,0,1,1]", class_name="face", confidence=0.9,
            source="import", proposed_name="Ana", proposal_basis="unambiguous",
            created_at=datetime(2026, 8, 5, 12, 0),
        ))
        s.commit()

    body = client.get(f"/library/import/done?source_id={source_id}").text
    assert "faces found" in body
    assert "names proposed" in body
    assert "<span>indexed</span>" not in body
    assert "<span>pending</span>" not in body


def test_the_done_page_says_proposals_are_not_confirmations(env, spy):
    client, take, _ = env
    source_id = start(client, take, "takeout")
    body = client.get(f"/library/import/done?source_id={source_id}").text
    assert "Proposals are guesses" in body
    assert "/training" in body


def test_a_directory_source_keeps_the_index_status_counts(env, spy):
    client, _, plain = env
    source_id = start(client, plain, "directory")
    body = client.get(f"/library/import/done?source_id={source_id}").text
    assert "<span>indexed</span>" in body
    assert "faces found" not in body
