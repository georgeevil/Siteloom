"""The library's "not indexed yet" banner (CLD-126).

Two-phase indexing means a scanned library legitimately renders as a
grid of blank placeholders — `thumb_path` is written by the second pass
— and the screen used to say nothing at all about it, which reads as a
failed import. What is asserted here is the saying-so, and the one rule
the action must not break: which pass a source gets is decided by its
kind, because running the plain pass over a Takeout archive writes the
face annotations that make `takeout import` skip those items later.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import SiteConfig, StorageConfig
from siteloom.store import (
    LibraryItem,
    LibrarySource,
    OperationRun,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import library_routes
from siteloom.web.app import create_app

TS = datetime(2026, 8, 15, 9, 0, 0)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/lib.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    client = TestClient(create_app(config))
    yield client, Session, tmp_path
    library_routes._import_state["thread"] = None


def _source(Session, tmp_path, *, name="photos", kind="directory", **statuses):
    """A source with `n` items in each named status."""
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    with Session() as session:
        source = LibrarySource(path=str(path), name=name, kind=kind, added_at=TS)
        session.add(source)
        session.flush()
        for status, count in statuses.items():
            for n in range(count):
                session.add(
                    LibraryItem(
                        source_id=source.id,
                        path=str(path / f"{status}{n}.jpg"),
                        kind="image",
                        status=status,
                        mtime=TS,
                    )
                )
        session.commit()
        return source.id


class _Running:
    def is_alive(self):
        return True


# -- what the backlog says --------------------------------------------------


def test_pending_and_failed_are_counted_apart(env):
    _, Session, tmp_path = env
    sid = _source(Session, tmp_path, pending=3, failed=2, indexed=1)
    with Session() as session:
        backlog = library_routes.index_backlog(session)
    assert backlog["pending"] == 3
    # Never folded into pending: nothing picks a failed item up again, so
    # counting them together promises a run that will not happen.
    assert backlog["failed"] == 2
    assert backlog["target"] == sid


def test_a_fully_indexed_library_has_no_backlog(env):
    _, Session, tmp_path = env
    _source(Session, tmp_path, indexed=4, skipped=1)
    with Session() as session:
        assert library_routes.index_backlog(session)["pending"] == 0
        assert library_routes.index_backlog(session)["sources"] == []


def test_two_waiting_sources_leave_no_target_to_guess_at(env):
    """A Takeout archive and a directory need different passes, so one
    button cannot serve both — the banner offers a choice instead."""
    _, Session, tmp_path = env
    _source(Session, tmp_path, name="trip", pending=2)
    _source(Session, tmp_path, name="takeout", kind="takeout", pending=5)
    with Session() as session:
        backlog = library_routes.index_backlog(session)
    assert backlog["target"] is None
    assert {s["name"] for s in backlog["sources"]} == {"trip", "takeout"}
    assert backlog["pending"] == 7


def test_filtering_to_a_source_makes_it_the_target(env):
    _, Session, tmp_path = env
    _source(Session, tmp_path, name="trip", pending=2)
    wanted = _source(Session, tmp_path, name="takeout", kind="takeout", pending=5)
    with Session() as session:
        backlog = library_routes.index_backlog(session, wanted)
    assert backlog["target"] == wanted
    assert backlog["pending"] == 5  # the filter's backlog, not the library's


# -- what the screen shows --------------------------------------------------


def test_the_grid_explains_why_it_is_blank(env):
    client, Session, tmp_path = env
    _source(Session, tmp_path, pending=7)
    body = client.get("/library").text
    assert "not indexed yet" in body
    assert "Start indexing" in body
    assert 'action="/library/index"' in body


def test_an_indexed_library_shows_no_banner(env):
    client, Session, tmp_path = env
    _source(Session, tmp_path, indexed=3)
    body = client.get("/library").text
    assert "not indexed yet" not in body
    assert "Start indexing" not in body


def test_failed_items_get_their_own_line_and_an_explicit_retry(env):
    client, Session, tmp_path = env
    _source(Session, tmp_path, failed=2, indexed=1)
    body = client.get("/library").text
    assert "failed earlier" in body
    assert 'name="retry_failed" value="1"' in body


def test_failures_are_still_named_when_no_one_button_can_retry_them(env):
    """A failed item nobody mentions is indistinguishable from one that
    was never there — which is the reason `failed` is counted apart from
    `pending` in the first place. The button can be missing; the count
    may not be."""
    client, Session, tmp_path = env
    _source(Session, tmp_path, name="trip", failed=2)
    _source(Session, tmp_path, name="takeout", kind="takeout", pending=5)
    body = client.get("/library").text
    assert "failed earlier" in body
    assert "Filter to one source" in body
    assert 'name="retry_failed" value="1"' not in body


# -- starting the run -------------------------------------------------------


def _wait_for_run(Session, kind, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session() as session:
            run = session.query(OperationRun).filter_by(kind=kind).first()
            if run is not None:
                return run
        time.sleep(0.05)
    return None


def test_starting_an_index_run_lands_on_the_jobs_page(env):
    client, Session, tmp_path = env
    sid = _source(Session, tmp_path, indexed=1)  # nothing to do: finishes at once
    response = client.post(
        "/library/index", data={"source_id": sid}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs"
    # The platform owns progress: the run is an OperationRun row, which is
    # what /jobs already renders and what a second terminal can watch.
    assert _wait_for_run(Session, "library-index") is not None


def test_a_takeout_source_gets_the_importer_not_the_plain_pass(env):
    """CLD-92's rule, now reachable from a second button. Indexing a
    Takeout archive with the plain pass is not slow, it is lossy: it
    writes the face annotations that make a later `takeout import` skip
    the item, and the name proposals never happen."""
    client, Session, tmp_path = env
    sid = _source(Session, tmp_path, name="takeout", kind="takeout", indexed=1)
    assert (
        client.post(
            "/library/index", data={"source_id": sid}, follow_redirects=False
        ).status_code
        == 303
    )
    assert _wait_for_run(Session, "takeout-import") is not None


def test_a_second_run_is_refused_where_the_first_one_is_visible(env):
    """A person clicked a form, so the answer is a page — and it is the
    page that answers their next question: then what *is* running?"""
    client, Session, tmp_path = env
    sid = _source(Session, tmp_path, pending=1)
    library_routes._import_state["thread"] = _Running()
    response = client.post(
        "/library/index", data={"source_id": sid}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs?notice=")
    assert "already+going" in response.headers["location"]
    # Refused means refused: no second run was started behind it.
    assert _wait_for_run(Session, "library-index", timeout=0.3) is None


def test_the_wizard_and_the_banner_share_one_guard(env):
    """Same job behind two buttons — two passes would fight over the same
    pending rows and the one embedded vector store."""
    client, Session, tmp_path = env
    sid = _source(Session, tmp_path, pending=1)
    library_routes._import_state["thread"] = _Running()
    response = client.post(
        "/library/import/index", data={"source_id": sid}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs?notice=")


def test_an_unknown_source_is_a_404(env):
    client, _, _ = env
    assert client.post("/library/index", data={"source_id": 999}).status_code == 404
