from __future__ import annotations

from datetime import datetime, timedelta
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
    """The triage row's status chip tracks the verdicts recorded on it."""
    assert "st-chip st-new" in webenv.client.get("/").text

    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "confirmed"})
    assert "st-chip st-reviewing" in webenv.client.get("/").text

    # A recorded miss outranks a confirmed claim — the event still needs someone.
    webenv.client.post("/events/1/missed", data={"missed": "1"})
    assert "st-chip st-flagged" in webenv.client.get("/").text


def test_clearing_an_event_takes_it_out_of_needs_review(webenv):
    """Sign-off is the only thing that empties the queue, and it reverses."""
    assert "st-chip st-new" in webenv.client.get("/", params={"needs_review": 1}).text

    webenv.client.post("/events/1/review", data={"reviewed": "1"})
    assert "No events match" in webenv.client.get("/", params={"needs_review": 1}).text
    assert "st-chip st-cleared" in webenv.client.get("/").text

    webenv.client.post("/events/1/review", data={"reviewed": "0"})
    assert "st-chip st-new" in webenv.client.get("/", params={"needs_review": 1}).text


def test_clearing_leaves_identity_verdicts_alone(webenv):
    """Clearing says "needs nothing further", not "every claim was right"."""
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "wrong"})
    webenv.client.post("/events/1/review", data={"reviewed": "1"})
    with webenv.Session() as session:
        assert session.get(EventIdentity, 1).verdict == "wrong"


def test_review_redirect_cannot_leave_the_site(webenv):
    """next_url comes from the page, so it is attacker-supplied."""
    r = webenv.client.post(
        "/events/1/review",
        data={"reviewed": "1", "next_url": "//evil.example/x"},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/events/1"


def test_unmatched_chip_selects_events_with_no_identity(webenv):
    # The seeded event has an identity link, so Unmatched must exclude it.
    assert "No events match" in webenv.client.get("/", params={"unmatched": 1}).text


def test_kind_chips_widen_rather_than_narrow(webenv):
    """People and Vehicles are alternatives — ticking both shows both."""
    assert "No events match" in webenv.client.get("/", params={"kind": "people"}).text
    both = webenv.client.get("/", params=[("kind", "people"), ("kind", "vehicles")])
    assert "car" in both.text


def test_rail_renders_for_a_selected_event(webenv):
    page = webenv.client.get("/", params={"selected": 1}).text
    assert "triage-rail" in page
    assert "Clear event" in page
    assert webenv.client.get("/events/999/rail").status_code == 404


@pytest.fixture
def deep_link_env(tmp_path):
    """More events than one page, with the interesting one the OLDEST —
    the shape a search deep-link lands in (`/?selected=<id>` where the
    event is not on the loaded page)."""
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/deep.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    base = datetime(2026, 8, 6, 10, 0, 0)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        session.flush()
        old = Event(
            camera_id="cam1",
            track_id=1,
            class_name="truck",
            first_seen=base,
            last_seen=base,
            detection_count=1,
        )
        session.add(old)
        session.flush()
        # 52 newer events fill page one (50 per page) and push `old` off it.
        for i in range(2, 54):
            session.add(
                Event(
                    camera_id="cam1",
                    track_id=i,
                    class_name="car",
                    first_seen=base + timedelta(hours=i),
                    last_seen=base + timedelta(hours=i),
                    detection_count=1,
                )
            )
        session.commit()
        old_id = old.id
    return SimpleNamespace(client=TestClient(create_app(config)), old_id=old_id)


def test_rail_deep_link_to_an_event_off_the_loaded_page(deep_link_env):
    """Regression: the rail used to 500 here with a DetachedInstanceError
    — the selected event was fetched without eager options and the
    template lazy-loaded its camera after the session had closed."""
    r = deep_link_env.client.get("/", params={"selected": deep_link_env.old_id})
    assert r.status_code == 200
    assert "triage-rail" in r.text
    assert "Cam One" in r.text
    # The standalone fragment endpoint serves the same off-page event.
    frag = deep_link_env.client.get(f"/events/{deep_link_env.old_id}/rail")
    assert frag.status_code == 200
    assert "Cam One" in frag.text


@pytest.fixture
def gated_env(tmp_path):
    """One significant event and one ephemeral fragment on the same camera."""
    config = SiteConfig(
        site_id="test-site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/gated.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        session.add(
            Event(
                camera_id="cam1",
                track_id=1,
                class_name="person",
                first_seen=datetime(2026, 8, 5, 12, 0, 0),
                last_seen=datetime(2026, 8, 5, 12, 0, 30),
                detection_count=12,
                best_confidence=0.9,
                significant=True,
            )
        )
        session.add(
            Event(
                camera_id="cam1",
                track_id=None,
                class_name="dog",
                first_seen=datetime(2026, 8, 5, 12, 1, 0),
                last_seen=datetime(2026, 8, 5, 12, 1, 1),
                detection_count=1,
                best_confidence=0.45,
                significant=False,
            )
        )
        session.commit()
    return SimpleNamespace(client=TestClient(create_app(config)), Session=Session)


def test_default_view_hides_ephemeral_events(gated_env):
    r = gated_env.client.get("/")
    assert r.status_code == 200
    assert 'data-event="1"' in r.text  # significant person event
    assert 'data-event="2"' not in r.text  # ephemeral dog fragment
    # The count line links to the hidden events rather than losing them.
    assert "1 ephemeral hidden" in r.text


def test_ephemeral_chip_reveals_gated_events(gated_env):
    r = gated_env.client.get("/", params={"show_ephemeral": "1"})
    assert 'data-event="1"' in r.text
    assert 'data-event="2"' in r.text
    assert "ephemeral hidden" not in r.text


def test_min_conf_and_min_count_filter_events(gated_env):
    r = gated_env.client.get(
        "/", params={"show_ephemeral": "1", "min_conf": "0.8"}
    )
    assert 'data-event="1"' in r.text and 'data-event="2"' not in r.text
    r = gated_env.client.get(
        "/", params={"show_ephemeral": "1", "min_count": "5"}
    )
    assert 'data-event="1"' in r.text and 'data-event="2"' not in r.text


def test_triage_url_round_trips_new_filters(gated_env):
    """Toggling any chip must preserve the ephemeral flag and numeric
    filters, like every other filter (the _triage_url contract)."""
    r = gated_env.client.get(
        "/",
        params={
            "show_ephemeral": "1",
            "min_conf": "0.4",
            "min_count": "1",
            "camera": "cam1",
        },
    )
    assert r.status_code == 200
    # The Needs-review chip link keeps all three new params.
    assert "show_ephemeral=1" in r.text
    assert "min_conf=0.4" in r.text
    assert "min_count=1" in r.text
