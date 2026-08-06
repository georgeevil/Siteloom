"""Global search and the SSE job stream (CLD-26/27 decisions)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    LibraryItem,
    LibrarySource,
    OperationRun,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 6, 9, 0, 0)


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="north", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="north", site_id="t", name="North Gate"))
        identity = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Ardent Freight",
            plate="KJ-4471",
            first_seen=TS,
            last_seen=TS,
        )
        s.add(identity)
        s.flush()
        event = Event(
            camera_id="north",
            track_id=1,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=3,
        )
        s.add(event)
        s.flush()
        s.add(
            EventIdentity(
                event_id=event.id, identity_id=identity.id, similarity=0.9
            )
        )
        s.add(LibrarySource(id=1, path="/a", name="A", added_at=TS))
        s.add(
            LibraryItem(
                source_id=1,
                path="/a/ardent-dock.jpg",
                kind="image",
                status="indexed",
                mtime=TS,
            )
        )
        s.add(
            OperationRun(
                kind="library-index",
                target="A",
                status="running",
                phase="index",
                current=5,
                total=10,
                started_at=TS,
                updated_at=TS,
            )
        )
        s.commit()
    return TestClient(create_app(config))


def test_search_finds_people_by_name(client):
    body = client.get("/search", params={"q": "ardent"}).text
    assert "Ardent Freight" in body
    assert "identity" in body


def test_search_finds_plates(client):
    body = client.get("/search", params={"q": "KJ-44"}).text
    assert "Ardent Freight" in body


def test_search_finds_events_through_the_matched_identity(client):
    """Searching a name must surface the visits, not just the person."""
    body = client.get("/search", params={"q": "ardent"}).text
    assert "Events · 1" in body


def test_search_by_camera_name_and_event_number(client):
    assert "Events · 1" in client.get("/search", params={"q": "North"}).text
    assert "Events · 1" in client.get("/search", params={"q": "1"}).text


def test_search_finds_library_files(client):
    body = client.get("/search", params={"q": "dock"}).text
    assert "ardent-dock.jpg" in body


def test_empty_query_explains_rather_than_errors(client):
    r = client.get("/search")
    assert r.status_code == 200
    assert "Search matches" in r.text


def test_topbar_search_is_a_real_form(client):
    body = client.get("/").text
    assert 'action="/search"' in body
    assert 'name="q"' in body


def test_jobs_stream_emits_run_snapshots(client):
    """One tick of the SSE stream carries the same payload /api/jobs does."""
    with client.stream(
        "GET", "/api/jobs/stream", params={"updates": 1, "interval": 0.5}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    line = next(l for l in body.splitlines() if l.startswith("data: "))
    runs = json.loads(line[len("data: "):])
    assert runs and runs[0]["kind"] == "library-index"
    assert runs[0]["current"] == 5 and runs[0]["total"] == 10


def test_jobs_stream_clamps_the_interval(client):
    """A client asking for interval=0 must not make the server spin."""
    with client.stream(
        "GET", "/api/jobs/stream", params={"updates": 2, "interval": 0}
    ) as r:
        body = "".join(r.iter_text())
    assert body.count("data: ") == 2  # bounded, and it terminated
