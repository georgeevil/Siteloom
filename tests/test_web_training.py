"""The Training data screen's filters, grouping and source rail (CLD-24).

These cover the query layer behind the three panes. The crop filters are
the part worth guarding: they decide what an operator sees as "still to
do", and a wrong predicate silently hides work rather than failing.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app
from siteloom.web.library_routes import _source_progress

TS = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(LibrarySource(id=1, path="/archive/a", name="Alpha", added_at=TS))
        s.add(LibrarySource(id=2, path="/archive/b", name="Beta", added_at=TS))
        s.flush()
        # Alpha: 2 indexed, 1 pending, 1 failed. Beta: nothing indexed yet.
        for i, status in enumerate(["indexed", "indexed", "pending", "failed"]):
            s.add(
                LibraryItem(
                    id=i + 1,
                    source_id=1,
                    path=f"/archive/a/{i}.jpg",
                    kind="image",
                    status=status,
                    mtime=TS,
                )
            )
        s.add(
            LibraryItem(
                id=9,
                source_id=2,
                path="/archive/b/0.jpg",
                kind="image",
                status="pending",
                mtime=TS,
            )
        )
        s.flush()
        # One annotation per state, all on Alpha's first item.
        states = [
            # (proposed_name, verified, rejected, enrolled)
            ("Ana", False, False, False),  # needs review
            ("Ana", True, False, True),  # verified + enrolled
            ("Bo", True, False, False),  # verified but never enrolled
            (None, False, True, False),  # rejected
        ]
        for i, (name, verified, rejected, enrolled) in enumerate(states):
            s.add(
                Annotation(
                    item_id=1,
                    bbox="[0,0,1,1]",
                    class_name="face",
                    confidence=0.9,
                    proposed_name=name,
                    verified=verified,
                    rejected=rejected,
                    enrolled=enrolled,
                    created_at=TS,
                )
            )
        # A vehicle crop, so the kind chips have something to separate.
        s.add(
            Annotation(
                item_id=2,
                bbox="[0,0,1,1]",
                class_name="car",
                confidence=0.8,
                created_at=TS,
            )
        )
        s.commit()
    return TestClient(create_app(config)), Session


def tiles(client, **params) -> int:
    r = client.get("/training", params=params)
    assert r.status_code == 200, r.text[:400]
    return r.text.count('class="tile"')


def test_each_state_filter_selects_only_its_own(env):
    client, _ = env
    assert tiles(client, show="needs_review") == 1
    assert tiles(client, show="verified") == 2
    assert tiles(client, show="rejected") == 1
    assert tiles(client, show="all") == 4


def test_unenrolled_finds_labels_the_system_cannot_see(env):
    """Verified with no vector — the failure mode enroll.py exists to fix."""
    client, _ = env
    r = client.get("/training", params={"show": "unenrolled"})
    assert r.text.count('class="tile"') == 1
    assert "Bo" in r.text  # the verified-but-unenrolled one
    assert "Ana" not in r.text.split('class="cropgrid"')[1]


def test_kind_chips_separate_faces_from_vehicles(env):
    client, _ = env
    assert tiles(client, show="all", kind="faces") == 4
    assert tiles(client, show="all", kind="vehicles") == 1
    assert tiles(client, show="all", kind="any") == 5


def test_source_filter_scopes_crops_to_that_source(env):
    client, _ = env
    assert tiles(client, show="all", kind="any", source_id=1) == 5
    # Beta has an item but no annotations on it.
    assert tiles(client, show="all", kind="any", source_id=2) == 0


def test_unnamed_crops_group_last_under_their_own_heading(env):
    """proposed_name is nullable, so grouping must not compare None to str."""
    client, _ = env
    body = client.get("/training", params={"show": "all"}).text
    headings = [
        chunk.split("</h3>")[0] for chunk in body.split("<h3>")[1:]
    ]
    assert headings == ["Ana", "Bo", "Unnamed"]


def test_flat_grouping_drops_the_headings(env):
    client, _ = env
    body = client.get("/training", params={"show": "all", "group": "flat"}).text
    assert "<h3>" not in body
    assert body.count('class="tile"') == 4


def test_source_rail_reports_failed_apart_from_pending(env):
    """Folding failed into pending would show a source as nearly done
    when part of it will never be processed without --retry-failed."""
    _, Session = env
    with Session() as s:
        rows = {r["source"].name: r for r in _source_progress(s)}
    alpha = rows["Alpha"]
    assert (alpha["indexed"], alpha["pending"], alpha["failed"]) == (2, 1, 1)
    assert alpha["total"] == 4
    assert alpha["percent"] == 50  # 2 of 4 done; the failed one is not done
    assert rows["Beta"]["percent"] == 0


def test_bad_filter_values_fall_back_rather_than_error(env):
    client, _ = env
    r = client.get("/training", params={"show": "../etc", "kind": "nope", "size": "xl"})
    assert r.status_code == 200
    assert 'class="crops size-m"' in r.text
