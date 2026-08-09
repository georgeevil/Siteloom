"""Keyset paging as the console's three main lists actually serve it
(CLD-104): `/`, `/identities`, `/noise`.

`tests/test_paging.py` holds the mechanism to its contract. This file
holds the *screens* to theirs, over HTTP, because that is where the two
failures being fixed live:

* `/` paginated with OFFSET, which is wrong on a live system rather than
  merely slow. The load-bearing test here therefore inserts rows
  *between* two fetches — on a static database OFFSET and keyset are
  indistinguishable, so a test that seeds and then reads proves nothing.
* `/identities` and `/noise` took `LIMIT 200` and said nothing about it,
  so the 201st row did not exist as far as the console was concerned.
  That is the more serious bug, and the tests for it just walk past 200.

Everything asserted about markup is asserted because a person depends on
it: the load-more control is a real `<a href>` so a no-JS client can walk
the list, and the cursor is absent from every other link so a URL an
operator copies opens at the top of that filtered set.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    Identity,
    NoiseEvent,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import app as app_mod
from siteloom.web.app import create_app
from siteloom.web.paging import encode_cursor

TS = datetime(2026, 8, 9, 12, 0, 0)


@pytest.fixture
def env(tmp_path):
    """An empty console plus a session factory to write behind its back —
    which is what a live ingest is, from the browser's point of view."""
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[
            CameraConfig(id="cam1", adapter="file", source="x"),
            CameraConfig(id="cam2", adapter="file", source="y"),
        ],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/scroll.db",
            media_dir=str(tmp_path / "media"),
        ),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        session.add(Camera(id="cam2", site_id="test-site", name="Cam Two"))
        session.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config)), Session=Session, config=config
    )


# -- seeding -------------------------------------------------------------


def add_events(env, count, *, camera="cam1", start=0, at=None):
    """`count` significant events, newest last, one second apart."""
    with env.Session() as session:
        for n in range(start, start + count):
            when = at or TS + timedelta(seconds=n)
            session.add(
                Event(
                    camera_id=camera,
                    track_id=n,
                    class_name="person",
                    first_seen=when,
                    last_seen=when,
                    detection_count=5,
                    best_confidence=0.9,
                    confidence_sum=4.5,
                )
            )
        session.commit()


def add_identities(env, count):
    with env.Session() as session:
        for n in range(count):
            when = TS + timedelta(seconds=n)
            session.add(
                Identity(
                    identifier_key="face",
                    class_name="person",
                    first_seen=when,
                    last_seen=when,
                )
            )
        session.commit()


def add_noise(env, count):
    with env.Session() as session:
        for n in range(count):
            start = TS + timedelta(minutes=n)
            session.add(
                NoiseEvent(
                    camera_id="cam1",
                    start=start,
                    end=start + timedelta(seconds=90),
                    peak_db=-12.0,
                    mean_db=-20.0,
                )
            )
        session.commit()


# -- reading the rendered page -------------------------------------------

MORE = re.compile(r'<a class="more-btn" href="([^"]+)"')
EVENT_IDS = re.compile(r'data-event="(\d+)"')
CARD_IDS = re.compile(r'class="idcard[^"]*" href="[^"]*selected=(\d+)"')
NOISE_STARTS = re.compile(r"<div class=\"mono\">(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)</div>")


def more_url(body: str) -> str | None:
    """The load-more link, or None when the page says it is the end."""
    match = MORE.search(body)
    return html.unescape(match.group(1)) if match else None


def walk(client, url, ids, limit=50):
    """Every row the list serves, following the real load-more hrefs.

    Deliberately follows the anchor rather than constructing cursors: a
    no-JS client walks the list exactly this way, so if this loop works
    that path works.
    """
    seen: list[str] = []
    pages = 0
    while url is not None and pages < limit:
        body = client.get(url).text
        seen.extend(ids.findall(body))
        url = more_url(body)
        pages += 1
    return seen


# -- the reason this exists ----------------------------------------------


def test_rows_arriving_between_two_fetches_shift_nothing(env, monkeypatch):
    """The OFFSET bug, stated against the route.

    With `offset(page_size)` the second fetch would re-serve a row the
    operator already reviewed and skip one they never saw. Ingest writes
    between the two requests here, which is the only way to tell the two
    schemes apart — on a static database they agree.
    """
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 6)

    first = env.client.get("/").text
    page_one = EVENT_IDS.findall(first)
    assert len(page_one) == 3

    # Six more events land above the window while the operator reads.
    add_events(env, 6, start=100)

    second = env.client.get(more_url(first)).text
    page_two = EVENT_IDS.findall(second)

    assert not set(page_one) & set(page_two), "a row was served twice"
    # The three oldest events are exactly what should follow, and none of
    # the newcomers may push one of them off the end.
    assert len(page_two) == 3
    with env.Session() as session:
        oldest = [
            str(e.id)
            for e in session.query(Event).order_by(Event.last_seen).limit(3)
        ]
    assert set(page_two) == set(oldest), "a row was skipped"


def test_events_sharing_a_timestamp_are_all_served(env, monkeypatch):
    """Sampled cameras close many tracks on one frame. A cursor on
    `last_seen` alone would drop every row tying with the last delivered,
    which is silent data loss rather than a visible error."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 2)
    add_events(env, 7, at=TS)

    seen = walk(env.client, "/", EVENT_IDS)
    assert len(seen) == 7
    assert len(set(seen)) == 7


def test_walking_reaches_every_event_exactly_once(env, monkeypatch):
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 7)
    add_events(env, 45)
    seen = walk(env.client, "/", EVENT_IDS)
    assert len(seen) == 45 and len(set(seen)) == 45


# -- filters and deep links ----------------------------------------------


def test_filters_survive_a_load_more(env, monkeypatch):
    """A cursor continues *this* filtered list. Losing the camera on the
    second slice would quietly widen the queue an operator is working."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 8, camera="cam1")
    add_events(env, 8, camera="cam2", start=200)

    first = env.client.get("/?camera=cam1").text
    link = more_url(first)
    assert "camera=cam1" in link

    seen = walk(env.client, "/?camera=cam1", EVENT_IDS)
    assert len(seen) == 8
    with env.Session() as session:
        cameras = {
            session.get(Event, int(i)).camera_id for i in seen
        }
    assert cameras == {"cam1"}


def test_chip_and_row_links_never_carry_a_cursor(env, monkeypatch):
    """A cursor is a position in one client's scroll, not a filter. It
    must not reach a URL an operator could copy — a link pasted to a
    colleague has to open at the top of that filtered set."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 9)

    body = env.client.get(more_url(env.client.get("/").text)).text
    copyable = re.findall(r'class="(?:chip|row)[^"]*"\s+href="([^"]+)"', body)
    copyable += re.findall(r'href="([^"]+)"\s+data-event=', body)
    assert copyable, "no chip or row links found to check"
    for href in copyable:
        assert "after=" not in html.unescape(href), href


def test_a_selected_deep_link_still_renders_the_rail(env, monkeypatch):
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 9)
    with env.Session() as session:
        oldest = session.query(Event).order_by(Event.last_seen).first().id

    # Deep-linked to an event that is not on the first slice: the rail is
    # server-rendered from its own query, so it must not depend on the
    # event being in the loaded page.
    body = env.client.get(f"/?selected={oldest}").text
    assert 'id="triage-rail"' in body
    # ...and the selection rides along with the cursor on load-more, so
    # clicking it does not drop the rail on a no-JS client.
    assert f"selected={oldest}" in html.unescape(more_url(body))


def test_a_cursor_the_server_did_not_mint_is_refused(env):
    """Not silently ignored. Falling back to "no cursor" restarts the
    list from the top mid-scroll, and the operator sees rows they already
    worked appear again — that reads as corruption, not as an error."""
    add_events(env, 4)
    for bad in ("not-base64!!", "YWJj", encode_cursor(TS), encode_cursor(TS, 1, 2)):
        for path in ("/", "/identities", "/noise"):
            r = env.client.get(path, params={"after": bad})
            assert r.status_code == 400, (path, bad, r.status_code)


def test_an_empty_cursor_is_simply_the_top_of_the_list(env, monkeypatch):
    """`?after=` with nothing after it is a browser quirk, not an attack."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 4)
    r = env.client.get("/", params={"after": ""})
    assert r.status_code == 200
    assert len(EVENT_IDS.findall(r.text)) == 3


# -- the control itself --------------------------------------------------


def test_load_more_is_a_real_link_a_no_js_client_can_follow(env, monkeypatch):
    """infinite.js only intercepts the click. If it never loads, the list
    must still be walkable."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 9)
    body = env.client.get("/").text
    assert "data-load-more" in body
    link = more_url(body)
    assert link and link.startswith("/") and "after=" in link
    # Following it is an ordinary navigation returning an ordinary page.
    r = env.client.get(link)
    assert r.status_code == 200
    assert "<html" in r.text


def test_end_of_list_says_so_and_does_not_look_like_an_empty_one(
    env, monkeypatch
):
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 3)
    body = env.client.get("/").text
    assert more_url(body) is None
    assert "End of the list" in body
    assert "3 events match these filters" in body
    # The empty state is a different message entirely.
    assert "No events match these filters" not in body


def test_an_empty_list_is_not_dressed_up_as_the_end_of_one(env):
    body = env.client.get("/").text
    assert "No events match these filters" in body
    assert "End of the list" not in body
    assert "data-load-more" not in body


def test_the_footer_carries_distinct_loading_end_and_error_slots(
    env, monkeypatch
):
    """Three states infinite.js switches between. If any of them shares
    an element with another, one of them renders as silence — which is
    what an empty list looks like."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 3)
    add_events(env, 9)
    body = env.client.get("/").text
    assert 'data-infinite-target="#event-rows"' in body
    assert 'id="event-rows"' in body
    for hook in ("data-load-more", "data-load-end", "data-load-error",
                 "data-load-sentinel"):
        assert hook in body, hook
    assert "more-loading" in body


# -- the screens that used to truncate in silence ------------------------


def test_identities_reaches_past_the_old_two_hundred_row_cap(env):
    """The 201st identity used to not exist as far as the console was
    concerned, and nothing on the page said so."""
    add_identities(env, 250)
    seen = walk(env.client, "/identities", CARD_IDS)
    assert len(seen) == 250
    assert len(set(seen)) == 250


def test_identities_counts_the_whole_filtered_set_not_the_slice(env):
    """The count used to be `len(rows)` — the page size — so it agreed
    with the truncation instead of exposing it."""
    add_identities(env, 250)
    body = env.client.get("/identities").text
    assert "250 of 250 identities" in body
    assert "200 of" not in body


def test_identities_filters_survive_a_load_more(env):
    add_identities(env, 80)
    body = env.client.get("/identities?unlabeled=1").text
    link = more_url(body)
    assert link is not None and "unlabeled=1" in link
    assert len(walk(env.client, "/identities?unlabeled=1", CARD_IDS)) == 80


def test_noise_reaches_past_the_old_two_hundred_row_cap(env):
    add_noise(env, 250)
    seen = walk(env.client, "/noise", NOISE_STARTS)
    assert len(seen) == 250
    assert len(set(seen)) == 250


def test_noise_states_its_real_total(env):
    """CLD-26's rule applied to a full table: a screen that cannot show
    the truth must not read as though it has."""
    add_noise(env, 250)
    body = env.client.get("/noise").text
    assert "250 recorded" in body
    assert more_url(body) is not None


def test_an_empty_noise_page_keeps_its_explanatory_state(env):
    """The empty states are the point of this page (CLD-26) and paging
    must not have quietly replaced them with a load-more footer."""
    body = env.client.get("/noise").text
    assert "data-load-more" not in body
    assert "Audio is on, but no camera is using it" in body
