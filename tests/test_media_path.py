"""Media paths must not depend on where the process was started (CLD-64).

A relative `storage.media_dir` used to be resolved against the CWD, so
the same site.yaml wrote crops into one tree and served them from
another — and since the /media containment check is anchored on that
same root, a wrong root is a wrong *security* answer, not only a 404.
The anchor is the config file's own directory, applied once in
load_config so every consumer inherits it.

Every test here launches the app from a directory that is *not* the
config's: one that stays in place cannot see this bug at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from siteloom.config import load_config, save_config
from siteloom.store import get_session, init_db, make_engine
from siteloom.web.app import create_app

CONFIG = """\
site_id: cwd-test
site_name: CWD Test
cameras:
  - id: cam1
    adapter: file
    source: x
storage:
  db_url: sqlite:///{db}
  media_dir: media
"""

CROP = b"\xff\xd8\xff-jpeg-ish"


@pytest.fixture
def site(tmp_path):
    """A config directory with a relative media_dir and one stored crop.

    `elsewhere` is where the process runs from — a different directory
    that also contains a `media` tree, so a CWD-relative resolution
    silently succeeds against the wrong files instead of failing loudly.
    """
    home = tmp_path / "site"
    home.mkdir()
    (home / "site.yaml").write_text(CONFIG.format(db=tmp_path / "cwd.db"))

    crop = home / "media" / "cam1" / "2026-08-09" / "120000_car_3.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(CROP)

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "media" / "cam1" / "2026-08-09").mkdir(parents=True)
    (elsewhere / "media" / "cam1" / "2026-08-09" / "120000_car_3.jpg").write_bytes(
        b"the wrong tree"
    )
    return home, elsewhere, crop


@pytest.fixture
def client(site, monkeypatch):
    home, elsewhere, _ = site
    monkeypatch.chdir(elsewhere)
    config = load_config(home / "site.yaml")
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    get_session(engine)
    return TestClient(create_app(config))


def test_media_dir_is_anchored_to_the_config_file(site, monkeypatch):
    """One place decides, so ingest, the indexer, doctor and the web
    layer cannot disagree about where media lives."""
    home, elsewhere, _ = site
    monkeypatch.chdir(elsewhere)
    config = load_config(home / "site.yaml")

    assert Path(config.storage.media_dir) == home / "media"
    # Same disease, same cure, for the other configured directories.
    assert Path(config.identity.vector_db_path) == home / "identity_db"
    assert Path(config.training.output_dir) == home / "training"


def test_absolute_media_dir_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "abs"
    home.mkdir()
    (home / "site.yaml").write_text(
        f"site_id: a\nstorage:\n  media_dir: {tmp_path}/pictures\n"
    )
    config = load_config(home / "site.yaml")
    assert config.storage.media_dir == f"{tmp_path}/pictures"


def test_media_serves_from_another_working_directory(client, site):
    """The load-bearing case: config in one directory, process in
    another, crop path as ingest stores it (absolute, media_dir prefix
    included)."""
    _, _, crop = site
    r = client.get("/media/" + str(crop))
    assert r.status_code == 200
    assert r.content == CROP  # not the identically-named decoy in the CWD


def test_media_serves_a_path_relative_to_media_root(client):
    """A path stored relative to media_dir resolves against media_root,
    not against wherever the server happens to be running."""
    r = client.get("/media/cam1/2026-08-09/120000_car_3.jpg")
    assert r.status_code == 200
    assert r.content == CROP


def test_media_serves_a_legacy_cwd_relative_path(client):
    """Rows written before the anchor carry the configured media_dir
    prefix on a relative path ("media/cam1/…"). Those must keep
    resolving — the fix cannot orphan an existing archive."""
    r = client.get("/media/media/cam1/2026-08-09/120000_car_3.jpg")
    assert r.status_code == 200
    assert r.content == CROP


def test_containment_still_refuses_escapes_from_another_cwd(client, tmp_path):
    """Anchoring relative paths must not open a traversal: the gate is
    still resolved-path containment, applied to every candidate."""
    outside = tmp_path / "elsewhere" / "secret.txt"
    outside.write_text("not yours")

    assert client.get("/media/%2e%2e%2f%2e%2e%2fetc/passwd").status_code == 404
    assert client.get("/media/media/../../secret.txt").status_code == 404
    assert client.get("/media/" + str(outside)).status_code == 404
    assert client.get("/media/etc/passwd").status_code == 404


def test_saving_a_config_keeps_the_operator_s_relative_path(site, monkeypatch):
    """Anchoring is a runtime reading, not an edit: a console save must
    not freeze this machine's absolute paths into a file that may be
    copied to the next host."""
    home, elsewhere, _ = site
    monkeypatch.chdir(elsewhere)
    path = home / "site.yaml"
    config = load_config(path)
    config.site_name = "Renamed"
    save_config(config)

    assert "media_dir: media" in path.read_text()
    assert str(home) not in path.read_text()
    # ...and reloading still anchors it.
    assert Path(load_config(path).storage.media_dir) == home / "media"


def test_an_edited_path_is_saved_as_edited(site, monkeypatch):
    home, elsewhere, _ = site
    monkeypatch.chdir(elsewhere)
    path = home / "site.yaml"
    config = load_config(path)
    config.storage.media_dir = "/mnt/nas/crops"
    save_config(config)

    assert "media_dir: /mnt/nas/crops" in path.read_text()
