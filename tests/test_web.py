from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import Camera, Detection, Event, get_session, init_db, make_engine
from siteloom.web.app import create_app


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/web.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=3,
            class_name="car",
            first_seen=datetime(2026, 8, 5, 12, 0, 0),
            last_seen=datetime(2026, 8, 5, 12, 0, 30),
            detection_count=1,
        )
        session.add(event)
        session.flush()
        session.add(
            Detection(
                event_id=event.id,
                timestamp=datetime(2026, 8, 5, 12, 0, 0),
                class_name="car",
                confidence=0.8,
                bbox="[1, 2, 3, 4]",
                zones='["driveway"]',
            )
        )
        session.commit()
    return TestClient(create_app(config))


def test_index_lists_events(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "car" in r.text
    assert "Cam One" in r.text


def test_index_class_filter(client):
    assert "car" in client.get("/", params={"class": "car"}).text
    assert "No events" in client.get("/", params={"class": "person"}).text


def test_event_detail(client):
    r = client.get("/events/1")
    assert r.status_code == 200
    assert "driveway" in r.text


def test_event_404(client):
    assert client.get("/events/999").status_code == 404


def test_media_path_confinement(client):
    assert client.get("/media/../../etc/passwd").status_code == 404
    assert client.get("/media/etc/passwd").status_code == 404
