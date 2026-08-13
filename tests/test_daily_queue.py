"""Today's queue on /training (CLD-8): the daily label-and-learn habit.

The queue is ~DAILY_QUEUE_TARGET borderline judgments pinned atop the
training screen — the crops where one label moves the model most
(Frigate's guidance: label the clear borderline crops, not the
90%-confident ones). The properties worth pinning are behavioural:

* Near-threshold beats high-confidence: an identity link whose
  similarity sits near its identifier's cutoff, and an annotation whose
  confidence sits in the middle band, are selected; the 0.99 match and
  the 0.95 crop are not.
* Deterministic within a day, rotating across days: reloading mid-session
  must not reshuffle the queue; tomorrow rotates through the borderline
  region rather than pinning the same skipped crops forever.
* Judged items leave and nothing slides in to replace them — a queue
  that refills as it is worked is how ten minutes becomes an hour.
* An empty queue is a *good* state and reads as one.
* The whole selection is SQL. It must render while ingest holds the
  vector store, so opening the store from this path is a failure even
  when the store happens to be free.

Nothing here needs weights, cameras or the vector store — query-layer
and markup tests over synthetic rows, like the rest of the web tests.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    Camera,
    Event,
    EventIdentity,
    Identity,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import library_routes
from siteloom.web.library_routes import (
    DAILY_QUEUE_TARGET,
    QUEUE_SIMILARITY_BAND,
    daily_queue,
)
from siteloom.web.app import create_app

#: Long before DAY, so every seeded row is eligible whatever the clock
#: says — rows created *on* the queue's day are tomorrow's queue.
TS = datetime(2026, 8, 1, 9, 0)
DAY = date(2026, 8, 10)


def make_env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/q.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return config, get_session(engine)


def add_link(
    s,
    row_id: int,
    similarity: float,
    matched_by: str = "visual",
    identifier: str = "vehicle",
):
    """One event + one identity claim on it, ids matching for legibility."""
    s.add(
        Event(
            id=row_id,
            camera_id="c",
            track_id=row_id,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=3,
            best_confidence=0.9,
            best_crop_path=f"c/2026-08-01/e{row_id}.jpg",
        )
    )
    s.add(
        EventIdentity(
            id=row_id,
            event_id=row_id,
            identity_id=1,
            identifier_key=identifier,
            similarity=similarity,
            matched_by=matched_by,
        )
    )


def add_annotation(s, row_id: int, **kwargs):
    fields = dict(
        id=row_id,
        item_id=1,
        bbox="[0,0,1,1]",
        class_name="face",
        confidence=0.55,
        source="auto",
        crop_path=f"library/crops/a{row_id}.jpg",
        created_at=TS,
    )
    fields.update(kwargs)
    s.add(Annotation(**fields))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """One of everything the selection must keep, and everything it must
    drop. The default vehicle threshold is 0.82, so 0.84 is borderline
    and 0.99 is the sure thing the queue exists to skip."""
    config, Session = make_env(tmp_path)
    monkeypatch.setattr(library_routes, "_queue_today", lambda: DAY)
    with Session() as s:
        s.add(Camera(id="c", site_id="t", name="Cam"))
        s.add(
            Identity(
                id=1,
                identifier_key="vehicle",
                class_name="car",
                label="Kestrel Sedan",
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.add(LibrarySource(id=1, path="/archive", name="A", added_at=TS))
        s.flush()
        s.add(
            LibraryItem(
                id=1,
                source_id=1,
                path="/archive/a.jpg",
                kind="image",
                status="indexed",
                mtime=TS,
            )
        )
        s.flush()
        add_link(s, 1, 0.84)  # borderline: |0.84 - 0.82| < band
        add_link(s, 2, 0.99)  # the sure thing
        add_link(s, 3, 0.83, matched_by="plate")  # synthetic similarity
        add_annotation(s, 1, confidence=0.9, proposed_name="Ana")  # proposal
        add_annotation(s, 2)  # mid-confidence crop
        add_annotation(s, 3, confidence=0.95)  # top of the range
        add_annotation(  # already judged, before today
            s,
            4,
            verified=True,
            verified_by="human",
            verified_at=TS,
        )
        add_annotation(s, 5, crop_path=None)  # nothing to look at
        s.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config)), Session=Session, config=config
    )


def queue_ids(body: str) -> list[tuple[str, int]]:
    """The queue's members in render order, as (kind, id) pairs."""
    return [
        (kind, int(row_id))
        for kind, row_id in re.findall(r'data-(link|annotation)="(\d+)"', body)
    ]


def entry_ids(queue: dict) -> list[tuple[str, int]]:
    return [
        (
            e["kind"],
            e["link"].id if e["kind"] == "link" else e["annotation"].id,
        )
        for e in queue["entries"]
    ]


# -- what "borderline" selects, and what it refuses -------------------------


def test_near_threshold_beats_high_confidence(env):
    """The selection rule itself: the borderline link, the name proposal
    and the mid-band crop are in, in that trust order; the 0.99 match,
    the plate match, the 0.95 crop, the already-judged crop and the
    crop with no image are all out."""
    with env.Session() as s:
        picked = entry_ids(daily_queue(s, env.config, DAY))
    assert picked == [("link", 1), ("annotation", 1), ("annotation", 2)]


def test_the_bands_are_the_configured_thresholds_not_a_pooled_scale(env):
    """Moving the identifier's threshold moves what counts as borderline:
    at 0.90 the 0.99 link is still sure, but 0.84 is no longer near the
    cutoff — nearness is per identifier, never a fixed score range."""
    env.config.identity.identifiers["vehicle"].threshold = (
        0.84 + QUEUE_SIMILARITY_BAND + 0.05
    )
    with env.Session() as s:
        picked = entry_ids(daily_queue(s, env.config, DAY))
    assert ("link", 1) not in picked


def test_deterministic_within_a_day(env):
    """Reloading must not reshuffle mid-session — the queue is the same
    list, in the same order, all day."""
    first = env.client.get("/training")
    second = env.client.get("/training")
    assert first.status_code == 200
    assert queue_ids(first.text) == queue_ids(second.text)
    assert queue_ids(first.text) == [
        ("link", 1),
        ("annotation", 1),
        ("annotation", 2),
    ]


def test_the_order_rotates_across_days(env):
    """Tomorrow's queue walks the borderline region in a different order,
    so a crop skipped today does not pin the queue forever. Membership
    here is identical (everything fits under the target); the rotation is
    the within-tier order."""
    with env.Session() as s:
        for row_id in range(10, 18):
            add_annotation(s, row_id)
        s.commit()
        one = entry_ids(daily_queue(s, env.config, date(2026, 8, 10)))
        two = entry_ids(daily_queue(s, env.config, date(2026, 8, 11)))
    assert sorted(one) == sorted(two)
    assert one != two


# -- judged items leave, and nothing slides in ------------------------------


def test_judged_items_leave_the_queue(env):
    """Both judgment paths are the console's existing endpoints — the
    review API for annotations, the verdict route for links — and either
    one takes its item out of the queue on the next render."""
    r = env.client.post(
        "/api/training/review",
        json={"decisions": [{"id": 2, "action": "reject"}]},
    )
    assert r.status_code == 200
    r = env.client.post(
        "/events/1/identity/1/verdict",
        data={"verdict": "confirmed", "next_url": "/training"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    body = env.client.get("/training").text
    assert queue_ids(body) == [("annotation", 1)]
    # The acknowledgement: one link verdict today. The rejection is not
    # counted — rejections carry no timestamp by schema design, and
    # under-counting is the honest direction for a number that must not
    # become a score.
    assert 'id="q-judged">1</span>' in body


def test_the_queue_does_not_refill_as_it_is_worked(tmp_path, monkeypatch):
    """With more borderline candidates than the target, judging a member
    must not slide another candidate in — membership is per-item, so the
    session converges to zero instead of becoming bottomless."""
    config, Session = make_env(tmp_path)
    monkeypatch.setattr(library_routes, "_queue_today", lambda: DAY)
    with Session() as s:
        s.add(Camera(id="c", site_id="t", name="Cam"))
        s.add(
            Identity(
                id=1,
                identifier_key="vehicle",
                class_name="car",
                first_seen=TS,
                last_seen=TS,
            )
        )
        for row_id in range(1, 31):
            add_link(s, row_id, 0.84)
        s.commit()
    client = TestClient(create_app(config))

    with Session() as s:
        before = {row_id for _, row_id in entry_ids(daily_queue(s, config, DAY))}
    # More candidates than fit: today's members are a strict subset.
    assert 0 < len(before) < 30
    victim = min(before)
    r = client.post(
        f"/events/{victim}/identity/{victim}/verdict",
        data={"verdict": "confirmed", "next_url": "/training"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with Session() as s:
        after = {row_id for _, row_id in entry_ids(daily_queue(s, config, DAY))}
    assert after == before - {victim}


def test_rows_created_today_wait_for_tomorrow(env):
    """A crop indexed mid-session is tomorrow's queue, never a mid-day
    arrival — the other half of "does not refill"."""
    with env.Session() as s:
        add_annotation(s, 30, created_at=datetime(2026, 8, 10, 9, 0))
        s.commit()
        picked = entry_ids(daily_queue(s, env.config, DAY))
    assert ("annotation", 30) not in picked
    with env.Session() as s:
        tomorrow = entry_ids(daily_queue(s, env.config, date(2026, 8, 11)))
    assert ("annotation", 30) in tomorrow


# -- the empty state is a good state ---------------------------------------


def test_nothing_borderline_reads_as_a_good_state(tmp_path, monkeypatch):
    config, Session = make_env(tmp_path)
    monkeypatch.setattr(library_routes, "_queue_today", lambda: DAY)
    client = TestClient(create_app(config))
    body = client.get("/training").text
    assert "Nothing borderline today" in body
    assert 'id="q-left">0</span>' in body


def test_a_cleared_queue_says_so_rather_than_reading_as_empty(env):
    """Worked to zero is the success state, and it must not render as the
    same words as "nothing qualified"."""
    with env.Session() as s:
        for link in s.query(EventIdentity).all():
            link.verdict = "confirmed"
            link.verdict_at = datetime(2026, 8, 10, 9, 30)
        for a in s.query(Annotation).all():
            a.rejected = True
        s.commit()
    body = env.client.get("/training").text
    assert "Queue cleared" in body
    assert "Nothing borderline today" not in body


# -- the queue is SQL, never the vector store -------------------------------


def test_the_queue_never_opens_the_vector_store(env, monkeypatch):
    """The whole point of the signal choice: the habit works *while
    ingest holds the store*. identity.enabled is on (the default) and the
    store is made unreachable — the section must still render, members
    and all. A signal that needs vectors is the wrong signal here."""
    assert env.config.identity.enabled is True

    def boom(*args, **kwargs):
        raise AssertionError("the daily queue must not touch the vector store")

    monkeypatch.setattr("siteloom.web.identity_ops.shared_store", boom)
    monkeypatch.setattr("siteloom.identity.get_shared_store", boom)
    r = env.client.get("/training")
    assert r.status_code == 200
    assert queue_ids(r.text) == [
        ("link", 1),
        ("annotation", 1),
        ("annotation", 2),
    ]


# -- module invariants ------------------------------------------------------


def test_the_day_hash_is_stable_across_processes():
    """sha256, not hash(): two workers (or a restart mid-morning) must
    agree on today's queue, and Python's hash() is salted per process."""
    value = library_routes._queue_hash(DAY, "crop", 7)
    assert value == library_routes._queue_hash(DAY, "crop", 7)
    assert 0.0 <= value < 1.0
    assert value != library_routes._queue_hash(date(2026, 8, 11), "crop", 7)
    assert value != library_routes._queue_hash(DAY, "link", 7)


def test_the_target_is_a_small_constant():
    """~20 is the ten-minute number the issue decided on; a config knob
    here would be an invitation to turn the habit into a shift."""
    assert DAILY_QUEUE_TARGET == 20
