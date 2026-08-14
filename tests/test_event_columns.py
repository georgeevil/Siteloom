"""The events list's column picker (CLD-108).

`/` orders by `Event.last_seen DESC` and used to render only
`first_seen` — sorted by a value the operator could not see, and the two
diverge exactly on the events de-fragmentation extended, which are the
events that matter. The fix is not a different sort; it is a column
picker with both timestamps (and Duration) on offer, defaulting to
today's composition plus Last seen.

What is held here, and why it is held server-side even though the picker
is client JavaScript:

* The offer is data (`EVENT_COLUMNS`, in the spirit of nav.py's NAV) and
  the template iterates it — so the header, the row cells, the picker's
  checkboxes and the hide rules cannot disagree about what is on offer.
* Every offered cell renders in every row, hidden or not: visibility is
  `hide-<key>` classes on the list container, which is the one design
  under which rows appended by infinite.js inherit the viewer's choice
  with no per-row state and no picker state in any URL.
* A no-JS client gets the default columns and loses nothing else; the
  picker control itself is simply absent (`hidden`).
* Duration reads as a duration, with the renderings decided in
  `duration_text` where they are testable.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import Camera, Event, get_session, init_db, make_engine
from siteloom.web import app as app_mod
from siteloom.web.app import (
    DEFAULT_HIDDEN_COLUMNS,
    EVENT_COLUMNS,
    create_app,
    duration_text,
)

TS = datetime(2026, 8, 9, 12, 0, 0)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/cols.db",
            media_dir=str(tmp_path / "media"),
        ),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        session.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config)), Session=Session, config=config
    )


def add_event(env, *, first=TS, last=None, count=5):
    with env.Session() as session:
        session.add(
            Event(
                camera_id="cam1",
                track_id=1,
                class_name="person",
                first_seen=first,
                last_seen=last or first + timedelta(minutes=14),
                detection_count=count,
                best_confidence=0.9,
                confidence_sum=4.5,
            )
        )
        session.commit()


ROW = re.compile(r'<a class="row rs-[^"]*"[\s\S]*?</a>')
LIST_CLASSES = re.compile(r'<div class="triage-list([^"]*)" id="triage-list"')
MORE = re.compile(r'<a class="more-btn" href="([^"]+)"')


def row_markup(body: str) -> str:
    match = ROW.search(body)
    assert match, "no event row rendered"
    return match.group(0)


# -- the offer table is the single source of truth -----------------------


def test_the_offer_is_data_with_both_timestamps_and_duration_on_it():
    keys = [c.key for c in EVENT_COLUMNS]
    assert len(keys) == len(set(keys)), "column keys must be unique"
    for want in ("arrived", "last-seen", "duration"):
        assert want in keys
    # Keys travel as CSS classes and localStorage tokens.
    for key in keys:
        assert re.fullmatch(r"[a-z][a-z0-9-]*", key), key


def test_the_default_set_is_todays_composition_plus_last_seen():
    on = {c.key for c in EVENT_COLUMNS if c.default}
    # Today's composition...
    assert {"arrived", "detection", "camera", "identity", "confidence"} <= on
    # ...plus the sort column, visible out of the box.
    assert "last-seen" in on
    # The additions beyond that are opt-in, so a first visit is unchanged.
    assert on == {"arrived", "last-seen", "detection", "camera", "identity", "confidence"}


def test_the_sort_column_is_last_seen_and_only_last_seen():
    sorted_keys = [c.key for c in EVENT_COLUMNS if c.sort]
    assert sorted_keys == ["last-seen"]


def test_default_hidden_is_exactly_the_non_default_columns():
    assert set(DEFAULT_HIDDEN_COLUMNS) == {
        c.key for c in EVENT_COLUMNS if not c.default
    }


# -- duration renders as a duration --------------------------------------


def test_duration_renderings_as_decided():
    def span(**kw):
        return duration_text(TS, TS + timedelta(**kw))

    # A single frame has no measured span; "0 s" would invent one.
    assert duration_text(TS, TS) == "< 1 s"
    assert span(seconds=0.4) == "< 1 s"
    assert span(seconds=38) == "38 s"
    assert span(seconds=59) == "59 s"
    assert span(minutes=1) == "1 min"
    assert span(minutes=14, seconds=40) == "14 min"
    assert span(minutes=59, seconds=59) == "59 min"
    assert span(hours=1) == "1 h 00"
    assert span(hours=2, minutes=5) == "2 h 05"
    assert span(hours=26, minutes=30) == "26 h 30"


def test_a_possibly_still_growing_event_carries_the_plus_marker():
    """The store has no open/closed bit, so "still active" is a horizon:
    last_seen within the track-link gap of now — the window in which
    ingest would still extend this very event."""
    first, last = TS, TS + timedelta(minutes=14)
    fresh = duration_text(first, last, now=last + timedelta(seconds=30), active_gap_s=120)
    assert fresh == "14 min +"
    stale = duration_text(first, last, now=last + timedelta(seconds=300), active_gap_s=120)
    assert stale == "14 min"
    # No horizon configured, or no clock supplied: never claim liveness.
    assert duration_text(first, last, now=last, active_gap_s=0) == "14 min"
    assert duration_text(first, last) == "14 min"
    # A just-started single-frame event is marked too, readably.
    assert duration_text(TS, TS, now=TS, active_gap_s=120) == "< 1 s +"


def test_the_page_marks_an_event_the_ingest_horizon_could_still_extend(env):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    add_event(env, first=now - timedelta(minutes=14), last=now)
    cell = re.search(r'class="r-time c-duration"[\s\S]*?</div>', env.client.get("/").text)
    assert cell and "14 min +" in cell.group(0)


# -- the server-rendered page --------------------------------------------


def test_every_offered_column_renders_a_cell_in_every_row(env):
    """The picker is client-side, so the template must serve any
    combination: all cells exist in the markup, chosen or not."""
    add_event(env)
    row = row_markup(env.client.get("/").text)
    for col in EVENT_COLUMNS:
        assert f"c-{col.key}" in row, col.key
    assert "c-status" in row  # fixed, not offered


def test_both_timestamps_actually_render_their_own_values(env):
    add_event(env, first=TS, last=TS + timedelta(minutes=14))
    row = row_markup(env.client.get("/").text)
    assert "12:00:00" in row  # arrived
    assert "12:14:00" in row  # last seen — the sort key, now visible


def test_the_default_hidden_columns_arrive_as_container_classes(env):
    """Hidden by class, not unrendered — the one mechanism the picker
    and infinite.js share. The classes sit on the list container so an
    appended slice inherits the choice for free."""
    add_event(env)
    body = env.client.get("/").text
    classes = LIST_CLASSES.search(body).group(1)
    assert set(classes.split()) == {f"hide-{k}" for k in DEFAULT_HIDDEN_COLUMNS}
    # And each offered column has its hide rule in the stylesheet.
    for col in EVENT_COLUMNS:
        assert f".triage-list.hide-{col.key} .c-{col.key}" in body


def test_the_header_iterates_the_offer_in_order_and_marks_the_sort(env):
    add_event(env)
    body = env.client.get("/").text
    head = re.search(r'<div class="row rowhead"[\s\S]*?\n      </div>', body).group(0)
    positions = [head.index(f"c-{col.key}") for col in EVENT_COLUMNS]
    assert positions == sorted(positions), "header order must follow the offer"
    # The sort affordance sits on the Last seen header cell and only there.
    sort_cells = re.findall(r'<div class="c-([a-z-]+) sortcol"', head)
    assert sort_cells == ["last-seen"]
    assert "sort-mark" in head
    assert head.count("sort-mark") == 1


def test_a_no_js_client_gets_the_default_columns_and_no_broken_picker(env):
    """Without JS the preference cannot be applied, so the control is
    simply absent (hidden); the default set still renders in full."""
    add_event(env)
    body = env.client.get("/").text
    picker = re.search(r"<details[^>]*id=\"column-picker\"[^>]*>", body).group(0)
    assert "hidden" in picker
    # The picker is not a form that posts anywhere.
    assert 'action=' not in picker
    # Every offered column is a checkbox, defaults pre-checked.
    pick = re.search(r'id="column-picker"[\s\S]*?</details>', body).group(0)
    for col in EVENT_COLUMNS:
        box = re.search(rf'value="{col.key}"( checked)?', pick)
        assert box, col.key
        assert bool(box.group(1)) == col.default, col.key


def test_no_link_on_the_page_carries_column_state(env, monkeypatch):
    """The choice is client-local, like the cursor (CLD-104): a chip, a
    row link or the load-more href pasted to a colleague must open with
    the default columns, not this browser's."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 2)
    for n in range(5):
        add_event(env, first=TS + timedelta(seconds=n))
    body = env.client.get("/").text
    for href in re.findall(r'href="([^"]+)"', body):
        href = html.unescape(href)
        assert "hide-" not in href and "columns" not in href, href


# -- load-more fragments carry the same structure ------------------------


def test_appended_slices_carry_the_same_cells_as_the_first(env, monkeypatch):
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 2)
    for n in range(5):
        add_event(env, first=TS + timedelta(seconds=n))
    first = env.client.get("/").text
    link = html.unescape(MORE.search(first).group(1))
    second = env.client.get(link).text

    def cell_shape(body):
        row = row_markup(body)
        return [key for key in re.findall(r'c-([a-z-]+)', row)]

    assert cell_shape(second) == cell_shape(first)
    # The fragment's container carries the same default hide classes, so
    # a no-JS walk shows the same columns page after page.
    assert (
        LIST_CLASSES.search(second).group(1) == LIST_CLASSES.search(first).group(1)
    )


def test_the_sort_stays_on_last_seen_and_the_cursor_still_works(env, monkeypatch):
    """The picker changes what is visible, never what orders the list:
    rows keep arriving newest-last-seen first across slices."""
    monkeypatch.setattr(app_mod, "EVENTS_PAGE", 2)
    for n in range(5):
        add_event(env, first=TS + timedelta(seconds=n))
    seen: list[str] = []
    url: str | None = "/"
    while url is not None:
        body = env.client.get(url).text
        seen.extend(re.findall(r'data-event="(\d+)"', body))
        match = MORE.search(body)
        url = html.unescape(match.group(1)) if match else None
    assert len(seen) == 5 and len(set(seen)) == 5
    with env.Session() as session:
        order = [
            str(e.id)
            for e in session.query(Event).order_by(
                Event.last_seen.desc(), Event.id.desc()
            )
        ]
    assert seen == order
