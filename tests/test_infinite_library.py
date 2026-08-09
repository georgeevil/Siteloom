"""Keyset paging on /library, /training and /audit (CLD-104).

Three screens, two different bugs.

`/library` and `/training` paginated with `OFFSET`. On a static database
OFFSET and keyset are indistinguishable, which is exactly why the offset
bug survived: it only shows itself when the set moves between two
fetches, and both of these sets move while they are being read. Indexing
registers and prunes library items; reviewing a crop takes it out of the
`needs_review` filter it was found under. So the load-bearing tests here
mutate the set *between* the two requests and assert that no row is
served twice and — the half OFFSET actually loses — that none is skipped.

`/audit` did not paginate at all: `AUDIT_LIMIT = 200`, and nothing on the
page said the 201st row existed. That is the worse bug of the two, and it
needs no concurrency to demonstrate: 205 rows, and the trail has to be
walkable to the end of them.

Nothing here needs model weights or a camera — these are query-layer and
markup tests over synthetic rows.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    AuditLog,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app
from siteloom.web.library_routes import LIBRARY_PAGE, TRAINING_PAGE
from siteloom.web.users_routes import AUDIT_PAGE

TS = datetime(2026, 8, 9, 9, 0)

#: The load-more control as the server renders it: an ordinary anchor
#: whose href carries the cursor. Matching on the markup rather than on a
#: helper is the point — the no-JS path is this element and nothing else.
_MORE = re.compile(r'<a class="lm-btn" href="([^"]+)"\s+data-load-more')


def more_href(body: str) -> str | None:
    """The next slice's URL, as a browser would read it, or None at the end."""
    match = _MORE.search(body)
    return html.unescape(match.group(1)) if match else None


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/s.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(LibrarySource(id=1, path="/archive", name="Alpha", added_at=TS))
        s.commit()
    return TestClient(create_app(config)), Session


# -- helpers ---------------------------------------------------------------


def add_items(Session, ids, status="indexed"):
    with Session() as s:
        for i in ids:
            s.add(
                LibraryItem(
                    id=i,
                    source_id=1,
                    path=f"/archive/item-{i}.jpg",
                    kind="image",
                    status=status,
                    mtime=TS,
                )
            )
        s.commit()


def add_crops(Session, ids, verified=False, rejected=False, name="Ana"):
    with Session() as s:
        if s.get(LibraryItem, 1) is None:
            s.add(
                LibraryItem(
                    id=1,
                    source_id=1,
                    path="/archive/item-1.jpg",
                    kind="image",
                    status="indexed",
                    mtime=TS,
                )
            )
            s.flush()
        for i in ids:
            s.add(
                Annotation(
                    id=i,
                    item_id=1,
                    bbox="[0,0,1,1]",
                    class_name="face",
                    confidence=0.9,
                    proposed_name=name,
                    verified=verified,
                    rejected=rejected,
                    created_at=TS,
                )
            )
        s.commit()


def add_audit(Session, count, username="ada", path="/events/1/review", start=1):
    with Session() as s:
        for n in range(start, start + count):
            s.add(
                AuditLog(
                    id=n,
                    at=TS + timedelta(seconds=n),
                    user_id=1,
                    username=username,
                    method="POST",
                    path=path,
                    status_code=303,
                )
            )
        s.commit()


def item_names(body: str) -> list[str]:
    return re.findall(r'<div class="mname">(item-\d+)\.jpg</div>', body)


def crop_ids(body: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="tile" data-id="(\d+)"', body)]


def audit_paths(body: str) -> list[str]:
    return re.findall(r'<div class="au-path">([^<]+)</div>', body)


# -- the reason this exists: the set moves between two fetches -------------


def test_library_items_pruned_mid_scroll_do_not_hide_the_tail(env):
    """The OFFSET bug on /library, stated as a test.

    Library items are ordered oldest-registered first, so what shifts the
    window is a row leaving it — a re-scan pruning files that are gone, or
    a source being cleaned up. Five rows the operator has already scrolled
    past disappear, and with `offset(60)` the five they had not reached
    yet fall off the end of the list and are never rendered at all.
    """
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 6))  # 65 items

    first = client.get("/library")
    seen = item_names(first.text)
    assert len(seen) == LIBRARY_PAGE

    with Session() as s:
        for i in range(1, 6):
            s.delete(s.get(LibraryItem, i))
        s.commit()

    second = client.get(more_href(first.text))
    assert second.status_code == 200
    rest = item_names(second.text)
    # The five the operator had not reached. OFFSET would have returned
    # none of them: 60 rows remain and it would ask for row 61 onwards.
    assert rest == [f"item-{i}" for i in range(61, 66)]
    assert not set(seen) & set(rest)


def test_reviewing_crops_mid_scroll_does_not_skip_the_ones_behind(env):
    """The same bug on /training, in the workflow that causes it.

    The queue is `show=needs_review`, and reviewing is what an operator is
    doing while they scroll: every crop they verify or reject leaves the
    filtered set, shifting everything below it up. With OFFSET the next
    slice starts as many rows too far down as they got through.
    """
    client, Session = env
    add_crops(Session, range(1, TRAINING_PAGE + 21))  # 68 unreviewed crops

    first = client.get("/training", params={"show": "needs_review", "group": "flat"})
    seen = crop_ids(first.text)
    assert len(seen) == TRAINING_PAGE

    with Session() as s:  # the operator rejects ten of them
        for i in range(1, 11):
            s.get(Annotation, i).rejected = True
        s.commit()

    second = client.get(more_href(first.text))
    rest = crop_ids(second.text)
    assert rest == list(range(TRAINING_PAGE + 1, TRAINING_PAGE + 21))
    assert not set(seen) & set(rest)


def test_audit_rows_arriving_mid_read_are_neither_repeated_nor_skipped(env):
    """/audit is written to by every other screen, so it is the list most
    certain to move while it is read: each slice an admin scrolls past is
    itself a request, and every mutation elsewhere in the console appends
    a row above the window."""
    client, Session = env
    add_audit(Session, AUDIT_PAGE + 5)

    first = client.get("/audit")
    seen = audit_paths(first.text)
    assert len(seen) == AUDIT_PAGE

    # Two mutations land while the admin reads the first slice.
    add_audit(Session, 2, path="/identities/9/label", start=AUDIT_PAGE + 100)

    second = client.get(more_href(first.text))
    rest = audit_paths(second.text)
    assert len(rest) == 5  # the five older rows, and nothing else
    assert "/identities/9/label" not in rest  # arrivals sit above the cursor


# -- the tie-break column --------------------------------------------------


def test_the_training_cursor_carries_the_column_that_ties(env):
    """`/training` sorts on `verified, id`, and `verified` is a boolean —
    it ties across the entire queue. A cursor carrying only that column
    would select `verified > False`, which is every row the operator has
    *not* asked for and none of the ones they have: the queue would empty
    itself after the first slice."""
    client, Session = env
    total = TRAINING_PAGE * 2 + 7
    add_crops(Session, range(1, total + 1))

    collected: list[int] = []
    url = "/training?show=needs_review&group=flat"
    while url:
        body = client.get(url).text
        collected.extend(crop_ids(body))
        url = more_href(body)

    assert sorted(collected) == list(range(1, total + 1))
    assert len(collected) == len(set(collected))


def test_a_verified_crop_still_appears_after_the_unverified_ones(env):
    """The order is unreviewed-first, and the cursor must be able to cross
    from one side of the boolean to the other without losing the rows on
    either side of the boundary."""
    client, Session = env
    add_crops(Session, range(1, TRAINING_PAGE), verified=False)
    add_crops(Session, range(100, 100 + 10), verified=True)

    collected: list[int] = []
    url = "/training?show=all&kind=faces&group=flat"
    while url:
        body = client.get(url).text
        collected.extend(crop_ids(body))
        url = more_href(body)

    assert collected == list(range(1, TRAINING_PAGE)) + list(range(100, 110))


# -- the silence /audit used to keep ---------------------------------------


def test_the_audit_trail_reaches_past_two_hundred_rows(env):
    """The acceptance criterion. `AUDIT_LIMIT = 200` meant the 201st row
    did not exist as far as the console was concerned, on the one screen
    an admin opens specifically to find out whether something happened."""
    client, Session = env
    add_audit(Session, 5, path="/the/oldest/thing")  # ids 1..5
    add_audit(Session, AUDIT_PAGE, path="/events/1/review", start=6)

    body = client.get("/audit").text
    assert "/the/oldest/thing" not in body  # still below the first slice
    assert f"<b>{AUDIT_PAGE + 5}</b> matching row" in body  # but counted

    older = client.get(more_href(body)).text
    assert older.count("/the/oldest/thing") == 5


def test_the_end_of_the_trail_says_so_rather_than_stopping(env):
    """Nothing-more-to-load and the-fetch-failed must not look alike, and
    neither may look like an empty list."""
    client, Session = env
    add_audit(Session, 3)

    body = client.get("/audit").text
    assert more_href(body) is None  # exhausted: no button at all
    assert 'data-state="end"' in body
    # ...and the end-of-list note is rendered visible, not hidden.
    assert re.search(r'data-load-end\s*>', body)


def test_a_full_first_slice_is_not_reported_as_exhausted(env):
    """Over-fetching by one is what answers "is there more" without a
    second COUNT that can disagree with the first."""
    client, Session = env
    add_audit(Session, AUDIT_PAGE)
    assert more_href(client.get("/audit").text) is None

    add_audit(Session, 1, start=AUDIT_PAGE + 1)
    assert more_href(client.get("/audit").text) is not None


# -- filters, and the URLs a cursor may not appear in -----------------------


@pytest.mark.parametrize(
    "url, params",
    [
        ("/library", {"status": "pending"}),
        ("/training", {"show": "needs_review", "kind": "faces", "group": "flat"}),
        ("/audit", {"path": "/events"}),
    ],
)
def test_a_filter_rides_along_on_the_load_more_link(env, url, params):
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 4), status="pending")
    add_items(Session, range(500, 505), status="indexed")
    add_crops(Session, range(1, TRAINING_PAGE + 4))
    add_audit(Session, AUDIT_PAGE + 3, path="/events/1/review")
    add_audit(Session, 4, path="/login", start=AUDIT_PAGE + 900)

    href = more_href(client.get(url, params=params).text)
    assert href is not None
    for key, value in params.items():
        assert f"{key}={value.replace('/', '%2F')}" in href or f"{key}={value}" in href


def test_the_library_filter_survives_being_scrolled_through(env):
    """The filter is the set; the cursor is a position in it. Following
    the load-more link must not widen one into the other."""
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 4), status="pending")
    add_items(Session, range(500, 600), status="indexed")

    first = client.get("/library", params={"status": "pending"})
    rest = client.get(more_href(first.text))
    names = item_names(rest.text)
    assert names == [f"item-{i}" for i in range(LIBRARY_PAGE + 1, LIBRARY_PAGE + 4)]
    assert "item-500" not in rest.text


def test_no_filter_link_carries_a_cursor(env):
    """A cursor describes one client's scroll. A link an operator can copy
    — a chip, a facet, a source — has to open at the top of the set it
    names, or a shared URL drops the recipient halfway down the sender's
    session."""
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 4))
    add_crops(Session, range(1, TRAINING_PAGE + 4))

    for url in ("/library", "/training"):
        body = client.get(url).text
        cursor = more_href(body).split("after=")[1]
        # Exactly one href in the page carries it: the load-more anchor.
        assert body.count(cursor) == 1


# -- the no-JS path --------------------------------------------------------


def test_the_load_more_control_is_a_real_link(env):
    """infinite.js only removes the clicking. If it never loads — a stale
    cache, a locked-down browser, an operator on a console with no
    scripting — the list still has to be walkable."""
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 4))

    body = client.get("/library").text
    href = more_href(body)
    assert href and href.startswith("/library?")
    # The enhancement contract: a container to append into, a sentinel to
    # observe, and a footer carrying the next cursor with the rows.
    assert 'data-infinite-target="#library-grid"' in body
    assert "data-load-sentinel" in body
    assert 'id="library-grid"' in body

    followed = client.get(href)
    assert followed.status_code == 200
    assert item_names(followed.text) == [
        f"item-{i}" for i in range(LIBRARY_PAGE + 1, LIBRARY_PAGE + 4)
    ]
    # Walkable is not quite usable: a page reached by following a cursor
    # offers the way back that only this client can need.
    assert 'href="/library">&uarr; back to the start' in followed.text
    assert "back to the start" not in body  # never on the first slice


def test_the_crop_grid_appends_into_the_container_it_names(env):
    """The target selector has to match a container that holds tiles and
    *not* the footer — appending a slice into the element the load-more
    control lives in would append the control to the list along with the
    rows. Grouped and flat lay out differently, so they name different
    containers, and the fetched page must agree with the one that asked."""
    client, Session = env
    add_crops(Session, range(1, TRAINING_PAGE + 4))

    flat = client.get("/training", params={"group": "flat"}).text
    assert 'data-infinite-target="#cropflat"' in flat
    assert '<div class="cropgrid" id="cropflat">' in flat

    grouped = client.get("/training", params={"group": "name"}).text
    assert 'data-infinite-target="#cropgrids"' in grouped
    # Same container in the slice that continues it, or nothing appends.
    assert 'id="cropgrids"' in client.get(more_href(grouped)).text


def test_the_fetched_page_carries_its_own_next_cursor(env):
    """What infinite.js parses out of a response: the target container and
    a footer describing what comes after the rows it just appended."""
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE * 2 + 4))

    second = client.get(more_href(client.get("/library").text)).text
    assert 'id="library-grid"' in second
    assert more_href(second) is not None  # a third slice is offered
    third = client.get(more_href(second)).text
    assert more_href(third) is None
    assert 'data-state="end"' in third


# -- cursors the server did not mint ---------------------------------------


@pytest.mark.parametrize("url", ["/library", "/training", "/audit"])
@pytest.mark.parametrize("token", ["not-base64!!", "YWJj", "Zm9vG2Jhcg"])
def test_a_malformed_cursor_is_refused_not_ignored(env, url, token):
    """Falling back to "no cursor" would restart the list at the top
    mid-scroll. The operator would see the first slice again and read it
    as duplicated rows, not as an error — so this is a 400, loudly."""
    client, Session = env
    add_items(Session, range(1, 5))
    add_crops(Session, range(1, 5))
    add_audit(Session, 4)

    r = client.get(url, params={"after": token})
    assert r.status_code == 400, r.text[:200]
    assert "cursor" in r.text


def test_a_cursor_from_another_list_is_refused(env):
    """/training sorts on two columns and /library on one, so their
    cursors are not interchangeable — and a token that decodes to the
    wrong shape must not be quietly truncated into a valid one."""
    client, Session = env
    add_items(Session, range(1, LIBRARY_PAGE + 4))
    add_crops(Session, range(1, TRAINING_PAGE + 4))

    library_cursor = more_href(client.get("/library").text).split("after=")[1]
    assert client.get("/training", params={"after": library_cursor}).status_code == 400
