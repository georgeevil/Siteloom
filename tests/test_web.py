from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
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


@pytest.fixture
def webenv(tmp_path):
    """A seeded app plus its session factory, for DB-level assertions."""
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
        identity = Identity(
            identifier_key="vehicle",
            class_name="car",
            first_seen=datetime(2026, 8, 5, 12, 0, 0),
            last_seen=datetime(2026, 8, 5, 12, 0, 30),
        )
        session.add(identity)
        session.flush()
        session.add(
            EventIdentity(event_id=event.id, identity_id=identity.id, similarity=0.9)
        )
        session.commit()
    return SimpleNamespace(client=TestClient(create_app(config)), Session=Session)


@pytest.fixture
def client(webenv):
    return webenv.client


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


def test_identity_verdict_confirm_and_clear(webenv):
    """A verdict persists with its timestamp; clearing removes both."""
    r = webenv.client.post(
        "/events/1/identity/1/verdict", data={"verdict": "confirmed"}
    )
    assert r.status_code == 200  # 303 followed to the event page
    with webenv.Session() as session:
        link = session.get(EventIdentity, 1)
        assert link.verdict == "confirmed"
        assert link.verdict_at is not None

    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "clear"})
    with webenv.Session() as session:
        link = session.get(EventIdentity, 1)
        assert link.verdict is None
        assert link.verdict_at is None


def test_identity_verdict_wrong_is_kept_not_deleted(webenv):
    """A wrong claim stays in the DB as a negative — never deleted."""
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "wrong"})
    with webenv.Session() as session:
        link = session.get(EventIdentity, 1)
        assert link is not None
        assert link.verdict == "wrong"


def test_identity_verdict_validation(webenv):
    assert (
        webenv.client.post(
            "/events/1/identity/1/verdict", data={"verdict": "maybe"}
        ).status_code
        == 400
    )
    # link 1 belongs to event 1 — reaching it through another event is a 404
    assert (
        webenv.client.post(
            "/events/999/identity/1/verdict", data={"verdict": "confirmed"}
        ).status_code
        == 404
    )


def test_missed_identity_toggle(webenv):
    webenv.client.post("/events/1/missed", data={"missed": "1"})
    with webenv.Session() as session:
        event = session.get(Event, 1)
        assert event.missed_identity is True
        assert event.missed_at is not None

    webenv.client.post("/events/1/missed", data={"missed": "0"})
    with webenv.Session() as session:
        event = session.get(Event, 1)
        assert event.missed_identity is False
        assert event.missed_at is None


def test_events_list_shows_review_state(webenv):
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "confirmed"})
    webenv.client.post("/events/1/missed", data={"missed": "1"})
    page = webenv.client.get("/").text
    assert "&#10003;1" in page  # confirmed count badge
    assert "badge missed" in page
