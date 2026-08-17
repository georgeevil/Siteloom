"""Drop-and-reindex: purge semantics and the /jobs/reindex endpoint."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.reindex import purge_window
from siteloom.store import (
    BackfillClip,
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

T0 = datetime(2026, 8, 6, 12, 0, 0)


@pytest.fixture
def env(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/reindex.db")
    init_db(engine)
    Session = get_session(engine)
    media = tmp_path / "media"
    (media / "cam1").mkdir(parents=True)
    crop = media / "cam1" / "crop1.jpg"
    crop.write_bytes(b"\xff\xd8jpg")
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Cam"))
        s.add(Camera(id="cam2", site_id="t", name="Other"))
        identity = Identity(
            identifier_key="person", class_name="person", first_seen=T0, last_seen=T0
        )
        s.add(identity)
        s.flush()
        # In-window event with a detection, a link, and a crop file.
        event = Event(
            camera_id="cam1", track_id=1, class_name="person",
            first_seen=T0, last_seen=T0 + timedelta(seconds=30),
            detection_count=1, best_crop_path="cam1/crop1.jpg",
        )
        s.add(event)
        s.flush()
        s.add(
            Detection(
                event_id=event.id, timestamp=T0, class_name="person",
                confidence=0.9, bbox="[1,2,3,4]", crop_path="cam1/crop1.jpg",
            )
        )
        s.add(EventIdentity(event_id=event.id, identity_id=identity.id))
        # Out-of-window event on the same camera: must survive.
        s.add(
            Event(
                camera_id="cam1", track_id=2, class_name="person",
                first_seen=T0 - timedelta(hours=10),
                last_seen=T0 - timedelta(hours=10),
                detection_count=1,
            )
        )
        # In-window event on another camera: must survive.
        s.add(
            Event(
                camera_id="cam2", track_id=3, class_name="person",
                first_seen=T0, last_seen=T0, detection_count=1,
            )
        )
        # Externally-owned (Frigate) event in window: must survive.
        s.add(
            Event(
                camera_id="cam1", external_id="fr-9", class_name="person",
                first_seen=T0, last_seen=T0, detection_count=1,
            )
        )
        # Backfill bookkeeping overlapping the window: must go, so a
        # re-scan re-registers the NVR windows.
        s.add(
            BackfillClip(
                camera_id="cam1", external_id="nvr-1",
                start=T0, end=T0 + timedelta(minutes=1),
            )
        )
        s.commit()
    return SimpleNamespace(Session=Session, media=media, crop=crop)


def test_purge_window_is_scoped_and_removes_crops(env):
    result = purge_window(
        env.Session,
        ["cam1"],
        T0 - timedelta(hours=1),
        T0 + timedelta(hours=1),
        env.media,
    )
    assert result.events == 1
    assert result.detections == 1
    assert result.links == 1
    assert result.clips == 1
    assert result.crop_files == 1
    assert not env.crop.exists()
    with env.Session() as s:
        remaining = {(e.camera_id, e.track_id, e.external_id) for e in s.query(Event)}
        # Out-of-window, other-camera, and external events all survive.
        assert remaining == {("cam1", 2, None), ("cam2", 3, None), ("cam1", None, "fr-9")}
        assert s.query(Identity).count() == 1  # identities are never purged
        assert s.query(BackfillClip).count() == 0


def test_purge_window_clears_a_cover_it_deleted_the_file_for(env):
    """Identity rows are never purged, but their cover can point into the
    window that just went (CLD-137). Leaving a row that is known-dangling
    for the renderer to hide is the same shape as clearing rows while
    leaving the media behind, which the factory reset exists to refuse —
    so the purge nulls the covers whose files it is about to delete.

    Event crops cannot reach this state: their rows go in the same pass.
    """
    with env.Session() as s:
        identity = s.query(Identity).one()
        identity.best_crop_path = "cam1/crop1.jpg"  # the crop about to go
        identity.cover_locked = True
        s.commit()

    purge_window(
        env.Session,
        ["cam1"],
        T0 - timedelta(hours=1),
        T0 + timedelta(hours=1),
        env.media,
    )

    assert not env.crop.exists()
    with env.Session() as s:
        identity = s.query(Identity).one()  # the identity itself survives
        assert identity.best_crop_path is None
        assert identity.cover_locked is False


def test_purge_window_leaves_a_cover_it_did_not_delete(env):
    """The other half: a cover pointing outside the purged window is
    still a picture of something that exists."""
    outside = env.media / "cam1" / "kept.jpg"
    outside.write_bytes(b"\xff\xd8jpg")
    with env.Session() as s:
        s.query(Identity).one().best_crop_path = "cam1/kept.jpg"
        s.commit()

    purge_window(
        env.Session, ["cam1"], T0 - timedelta(hours=1), T0 + timedelta(hours=1),
        env.media,
    )

    with env.Session() as s:
        assert s.query(Identity).one().best_crop_path == "cam1/kept.jpg"
    assert outside.exists()


def test_purge_window_is_idempotent(env):
    args = (env.Session, ["cam1"], T0 - timedelta(hours=1), T0 + timedelta(hours=1), env.media)
    purge_window(*args)
    again = purge_window(*args)
    assert (again.events, again.detections, again.links, again.clips) == (0, 0, 0, 0)


# -- /jobs/reindex endpoint -------------------------------------------------


def make_client(tmp_path, cameras):
    config = SiteConfig(
        site_id="t",
        cameras=cameras,
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/web.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return TestClient(create_app(config)), config


def test_reindex_rejects_without_unifi_cameras(tmp_path):
    client, _ = make_client(
        tmp_path, [CameraConfig(id="f", adapter="file", source="x")]
    )
    r = client.post("/jobs/reindex", data={"hours": "6"})
    assert r.status_code == 400


def test_reindex_rejects_concurrent_runs(tmp_path):
    from siteloom.web import library_routes

    client, _ = make_client(
        tmp_path, [CameraConfig(id="u", adapter="unifi", source="abc")]
    )
    stop = threading.Event()
    busy = threading.Thread(target=stop.wait)
    busy.start()
    library_routes._reindex_state["thread"] = busy
    try:
        r = client.post("/jobs/reindex", data={"hours": "6"})
        assert r.status_code == 409
    finally:
        stop.set()
        busy.join()
        library_routes._reindex_state["thread"] = None


def test_reindex_starts_background_job(tmp_path, monkeypatch):
    import siteloom.reindex as reindex_mod
    from siteloom import ingest as ingest_mod
    from siteloom.web import library_routes

    client, config = make_client(
        tmp_path, [CameraConfig(id="u", adapter="unifi", source="abc")]
    )
    calls = {}

    class FakeService:
        def __init__(self, cfg):
            calls["config"] = cfg
            self.config = cfg
            engine = make_engine(cfg.storage.db_url)
            self.Session = get_session(engine)

    def fake_reindex(service, cams, start, end, progress=None):
        calls["cams"] = [c.id for c in cams]
        calls["window_h"] = round((end - start).total_seconds() / 3600, 1)
        return reindex_mod.ReindexResult()

    monkeypatch.setattr(ingest_mod, "IngestService", FakeService)
    monkeypatch.setattr(reindex_mod, "reindex_window", fake_reindex)

    r = client.post(
        "/jobs/reindex", data={"hours": "6", "cameras": "u"}, follow_redirects=False
    )
    assert r.status_code == 303
    for _ in range(100):  # the work runs in a daemon thread
        thread = library_routes._reindex_state["thread"]
        if thread is None and "cams" in calls:
            break
        time.sleep(0.05)
    assert calls["cams"] == ["u"]
    assert calls["window_h"] == 6.0
    # The run is observable on the jobs board.
    from siteloom.store import OperationRun

    with FakeService(config).Session() as s:
        run = s.query(OperationRun).one()
        assert run.kind == "reindex"
        assert run.status == "complete"


def test_jobs_page_shows_reindex_form(tmp_path):
    client, _ = make_client(
        tmp_path, [CameraConfig(id="u", adapter="unifi", source="abc", name="Yard")]
    )
    r = client.get("/jobs")
    assert r.status_code == 200
    assert "Reindex from NVR" in r.text
    assert "Yard" in r.text


def test_jobs_page_hides_form_without_unifi(tmp_path):
    client, _ = make_client(
        tmp_path, [CameraConfig(id="f", adapter="file", source="x")]
    )
    assert "Reindex from NVR" not in client.get("/jobs").text
