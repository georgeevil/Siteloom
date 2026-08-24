"""What a claim row may honestly say about itself (CLD-136).

Two columns on `EventIdentity` are shown to operators as though they were
measurements, and neither is one on every row:

* `similarity` is a hardcoded 1.0 on a plate match, and `_absorb_evidence`
  takes a `max`, so once a plate touches a row that 1.0 is permanent —
  the field is not a stale similarity, it is not a similarity at all.
* A manual link is written at 0.0, which rendered as `sim 0.00` beside a
  green tick: the console reporting that it is certain of a claim an
  operator made by hand, in the same words it uses for a match it is
  barely confident of.

And `matched_by` is upgraded in place by every later frame and every
merge, so it answers "the strongest evidence on this claim", never "how
this claim was first made". Nothing persists the latter, so the badge
must not imply it.

The rule lives in one pure function so the three screens that render it
cannot each decide differently, and so the branch that decides whether a
number is real is testable without rendering a page.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import (
    CameraConfig,
    CameraIdentityOverride,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.claims import link_claim
from siteloom.web.app import create_app
from siteloom.web.provenance import claim_display

TS = datetime(2026, 8, 12, 9, 0, 0)


def _claim(**overrides) -> EventIdentity:
    """A claim row with every column set, the way a flush would leave it."""
    fields = dict(
        event_id=1,
        identity_id=2,
        identifier_key="vehicle",
        similarity=0.0,
        hit_count=1,
        matched_by=None,
        learned_plate=False,
        verdict=None,
        verdict_at=None,
        unlinked_at=None,
    )
    fields.update(overrides)
    return EventIdentity(**fields)


# -- 1. the badge ----------------------------------------------------------


@pytest.mark.parametrize(
    ("matched_by", "badge"),
    [
        ("plate", "plate"),
        ("visual", "visual"),
        ("human", "manual"),  # the column's word is not the operator's
        (None, "new"),  # no existing identity cleared the bar; this claim made one
    ],
)
def test_the_badge_names_how_the_claim_is_evidenced(matched_by, badge):
    display = claim_display(_claim(matched_by=matched_by, similarity=0.9), 0.8)

    assert display.badge == badge
    assert display.title  # every badge explains itself somewhere


@pytest.mark.parametrize("matched_by", ["plate", "visual", "human"])
def test_the_badge_claims_evidence_not_first_contact(matched_by):
    """`matched_by` is upgraded in place — by `_absorb_evidence` on every
    later frame and by `fold_claim` on every merge — so a row that minted
    an identity and was re-matched later reads `visual`, and any row a
    plate ever touched reads `plate`. The tooltip has to describe the
    strongest evidence rather than the first, because the column cannot
    answer the second question and nothing else persists it."""
    display = claim_display(_claim(matched_by=matched_by, similarity=0.9), 0.8)

    assert "evidence" in display.title.lower()


# -- 2-5. when a number is real -------------------------------------------


@pytest.mark.parametrize("similarity", [1.0, 0.93])
def test_a_plate_row_never_shows_a_score(similarity):
    """Not merely "synthetic": `_match_plate` returns a hardcoded 1.0 and
    `_absorb_evidence` keeps the max, so a plate row's similarity is
    pinned and no later genuine cosine can ever surface through it.

    The 0.93 case is the one that makes this more than cosmetic — a
    plate row can carry a perfectly plausible-looking number, and it is
    no more a measurement than the 1.0 is.
    """
    display = claim_display(_claim(matched_by="plate", similarity=similarity), 0.82)

    assert display.badge == "plate"
    assert display.score is None


def test_a_visual_row_shows_its_score_and_the_bar():
    display = claim_display(_claim(matched_by="visual", similarity=0.83), 0.82)

    assert display.badge == "visual"
    assert display.score == 0.83
    assert display.threshold == 0.82


def test_a_pristine_manual_row_shows_no_number():
    """The defect in the issue's own words: a claim an operator made by
    hand carried `sim 0.00`, which reads as the machine reporting no
    confidence in something a human asserted outright."""
    display = claim_display(_claim(matched_by="human", similarity=0.0), 0.82)

    assert display.badge == "manual"
    assert display.score is None


def test_a_zero_score_is_withheld_whatever_the_badge_says():
    """0.0 is the "never measured" value on this column, so it is not a
    number to print next to any badge."""
    assert claim_display(_claim(matched_by="visual", similarity=0.0), 0.82).score is None
    assert claim_display(_claim(matched_by=None, similarity=0.0), 0.82).score is None


# -- 4. the shape where a manual row carries a real measurement ------------


@pytest.fixture
def attach_env(tmp_path):
    """A console with one car event and one vehicle identity to attach."""
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/claims.db", media_dir=str(tmp_path / "m")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="t", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=1,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=1,
        )
        identity = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Bo Truck",
            first_seen=TS,
            last_seen=TS,
        )
        session.add_all([event, identity])
        session.commit()
        ids = SimpleNamespace(event=event.id, identity=identity.id)
    return SimpleNamespace(
        client=TestClient(create_app(config)), Session=Session, ids=ids
    )


def test_a_manual_row_that_absorbed_a_real_score_shows_it(attach_env):
    """The composition the CLD-151 review surfaced, built the way it
    actually happens rather than by writing the columns by hand.

    An operator attaches the identity (`similarity=0.0`, `human`), and an
    ingest frame arrives *afterwards*: `link_claim` finds the standing
    row and records the sighting, so `similarity` becomes 0.97 while
    `matched_by` stays `human` — `_MATCH_RANK` ranks "visual" and "human"
    equal, and only a strictly greater rank wins.

    So the row reads `manual` and carries a genuine cosine. Keying the
    number to the badge would hide a true measurement on exactly the
    rows an operator is most likely auditing. Built through the real
    path so it fails if that tie rule ever changes.
    """
    r = attach_env.client.post(
        f"/events/{attach_env.ids.event}/identity",
        data={"identity_id": attach_env.ids.identity, "enroll": "0"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with attach_env.Session() as session:
        link, created = link_claim(
            session,
            event_id=attach_env.ids.event,
            identity_id=attach_env.ids.identity,
            identifier_key="vehicle",
            similarity=0.97,
            matched_by="visual",
        )
        session.commit()
        assert created is False  # it landed on the operator's row
        # The shape itself, before asking what it renders as.
        assert (link.matched_by, link.similarity) == ("human", 0.97)

        display = claim_display(link, 0.82)

    assert display.badge == "manual"
    assert display.score == 0.97
    # The gloss earns its place here: the operator asserted the claim,
    # the machine contributed the number.
    assert display.title


# -- 6. the bar is the one in force now ------------------------------------


def _threshold(**camera_kwargs) -> float | None:
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x", **camera_kwargs)],
    )
    return config.identity.threshold_for("vehicle", config.cameras[0])


def test_the_bar_is_the_effective_threshold_for_this_camera():
    """Per-camera override first (CLD-39), then the identifier's
    site-wide value — the same resolution ingest used, not a second
    reading of the config."""
    overridden = _threshold(
        identity=CameraIdentityOverride(thresholds={"vehicle": 0.9})
    )
    assert overridden == 0.9
    assert claim_display(
        _claim(matched_by="visual", similarity=0.95), overridden
    ).threshold == 0.9

    assert _threshold() == 0.82  # the identifier's own value


def test_an_unconfigured_identifier_shows_the_score_with_no_bar():
    """An auto-added class keeps its default in the registry, not in
    config, so there is no site-wide bar to quote. Absent is absent:
    inventing a number here would be the same lie as the plate 1.00."""
    config = SiteConfig(site_id="t")
    assert config.identity.threshold_for("deer") is None

    display = claim_display(_claim(matched_by="visual", similarity=0.71), None)

    assert display.score == 0.71
    assert display.threshold is None


def test_the_rule_is_read_off_the_row_not_the_event(attach_env):
    """A last structural check: two claims on one event can disagree
    about all of this, so the display is per claim."""
    attach_env.client.post(
        f"/events/{attach_env.ids.event}/identity",
        data={"identity_id": attach_env.ids.identity, "enroll": "0"},
        follow_redirects=False,
    )
    with attach_env.Session() as session:
        session.add(
            EventIdentity(
                event_id=attach_env.ids.event,
                identity_id=attach_env.ids.identity,
                identifier_key="vehicle",
                similarity=1.0,
                matched_by="plate",
                unlinked_at=TS,  # repudiated, so the index allows the pair
            )
        )
        session.commit()
        rows = session.scalars(
            select(EventIdentity).order_by(EventIdentity.id)
        ).all()

    displays = [claim_display(row, 0.82) for row in rows]
    assert {d.badge for d in displays} == {"manual", "plate"}
    assert [d.score for d in displays] == [None, None]
