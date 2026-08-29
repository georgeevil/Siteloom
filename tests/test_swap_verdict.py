"""Ruling on a machine-suspected ID swap (the occlusion layer's flag).

`ingest._check_swap` froze identity claims on a pair of events; the
endpoint here is how the freeze ends. The rules under test: the verdict
applies to the *pair*, confirm promotes both to `multi_subject` (the
operator mark the tracker corpus reads), and the note keeps the scores
plus the verdict either way — a rejected suspicion is a measured false
positive, the precision data auto-reconcile is gated on.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import Camera, Event, get_session, init_db, make_engine
from siteloom.web.app import create_app

TS = datetime(2026, 8, 19, 18, 6, 43)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/e.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.identity.vector_db_path = str(tmp_path / "vectors")
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam", site_id="t", name="Cam"))
        events = []
        for track in (7, 8):
            event = Event(
                camera_id="cam", track_id=track, class_name="person",
                first_seen=TS, last_seen=TS, detection_count=30,
                suspect_swap=True, suspect_swap_at=TS,
            )
            s.add(event)
            s.flush()
            events.append(event)
        for event, other in (events, reversed(events)):
            event.suspect_swap_note = json.dumps({
                "other_event": other.id,
                "crossed": ["a"], "a_own": 0.2, "a_other": 0.9,
                "b_own": 0.8, "b_other": 0.3, "min_margin": 0.05,
            })
        s.commit()
        ids = [e.id for e in events]
    client = TestClient(create_app(config))
    client.sessionmaker = Session
    return client, ids


def rows(client):
    with client.sessionmaker() as s:
        return {e.id: (e.suspect_swap, e.multi_subject, e.swap_note)
                for e in s.query(Event).all()}


def test_confirm_clears_the_pair_and_marks_both_multi_subject(env):
    client, (a, b) = env
    resp = client.post(f"/events/{a}/swap-verdict",
                       data={"verdict": "confirm"}, follow_redirects=False)
    assert resp.status_code == 303
    state = rows(client)
    for event_id in (a, b):
        suspect, multi, note = state[event_id]
        assert suspect is False
        assert multi is True          # the corpus mark, on both sides
        assert note["verdict"] == "confirm"
        assert note["a_other"] == 0.9  # the evidence survives the ruling


def test_reject_clears_the_pair_without_the_corpus_mark(env):
    client, (a, b) = env
    client.post(f"/events/{b}/swap-verdict",
                data={"verdict": "reject"}, follow_redirects=False)
    state = rows(client)
    for event_id in (a, b):
        suspect, multi, note = state[event_id]
        assert suspect is False
        assert multi is False
        assert note["verdict"] == "reject"


def test_a_verdict_needs_a_standing_suspicion(env):
    client, (a, _) = env
    client.post(f"/events/{a}/swap-verdict",
                data={"verdict": "reject"}, follow_redirects=False)
    again = client.post(f"/events/{a}/swap-verdict",
                        data={"verdict": "confirm"}, follow_redirects=False)
    assert again.status_code == 404


def test_only_the_two_verdicts_exist(env):
    client, (a, _) = env
    resp = client.post(f"/events/{a}/swap-verdict",
                       data={"verdict": "shrug"}, follow_redirects=False)
    assert resp.status_code == 400


def test_the_badge_and_verdict_controls_render_on_the_event_page(env):
    client, (a, b) = env
    body = client.get(f"/events/{a}").text
    assert "possible ID swap" in body
    assert f"/events/{a}/swap-verdict" in body
    assert f"/events/{b}" in body  # the counterpart is one click away
    # A ruled event drops the controls.
    client.post(f"/events/{a}/swap-verdict",
                data={"verdict": "reject"}, follow_redirects=False)
    assert "possible ID swap" not in client.get(f"/events/{a}").text
