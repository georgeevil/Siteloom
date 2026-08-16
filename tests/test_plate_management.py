"""Correcting a plate an operator cannot otherwise reach (CLD-134).

`Identity.plate` is write-once by design: a plate match beats visual
similarity outright (PRD §6.4), so a plate that changed itself would
move every future sighting of that number. The cost was that a plate
read wrong at mint had no way out of the console at all — the live site
carried an identity whose plate was `111111`, and judging the read
"wrong" left the identity untouched, by documented design.

So the operator becomes the one writer allowed to overwrite it. Three
things have to hold for that to be a fix rather than a button:

* **A clear has to stick.** The resolver's learn path fires on an *empty*
  plate, so clearing one is precisely the condition that invites the next
  sighting to re-learn the junk string. `plate_source == "operator"` is
  what stops it, and case 2 below is the whole issue in one test.
* **What is typed has to be what matching compares.** `Identity.plate` is
  matched against `normalize_plate`'d OCR output, so a typed `TYB-506`
  stored verbatim would never match anything — the edit would look like
  it worked and quietly do nothing.
* **The console must not manufacture ambiguity.** Two identities sharing
  a plate makes the plate-first lookup pick between them arbitrarily, and
  an event with two vehicle claims cannot say which one a read belongs
  to. Both are refusals, not guesses.

No model weights: plates are rows, never embeddings, and the one visual
match needed is a hand-built unit vector.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.identity import get_shared_store
from siteloom.identity.resolver import IdentityResolver
from siteloom.store import (
    AuditLog,
    Camera,
    Event,
    EventIdentity,
    Identity,
    PlateRead,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.models import (
    PLATE_SOURCE_LEARNED,
    PLATE_SOURCE_MINT,
    PLATE_SOURCE_OPERATOR,
)
from siteloom.web import redaction
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role

TS = datetime(2026, 8, 10, 9, 0, 0)

#: The junk plate the live site actually carried, and the truth under it.
JUNK = "111111"
TRUTH = "TYB506"


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


VEHICLE_VECTOR = _unit(1.0, 0.0, 0.0, 0.0)


@pytest.fixture
def plate_env(tmp_path):
    """The live site's shape.

    One vehicle carrying a junk plate *read at mint* — the case no UI
    could reach, because no `learned_plate` link exists to revert — a
    second vehicle, a face identity, and three events carrying one, two
    and zero vehicle claims respectively. Event `two_claims` is event 30
    from the field: the reason apply-to-identity has to decline.
    """
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/plates.db", media_dir=str(tmp_path / "m")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    with Session() as session:
        session.add(Camera(id="cam1", site_id="t", name="Cam One"))
        events = [
            Event(
                camera_id="cam1",
                track_id=i,
                class_name="car",
                first_seen=TS,
                last_seen=TS,
                detection_count=1,
            )
            for i in range(3)
        ]
        session.add_all(events)
        session.flush()
        one_claim, two_claims, no_claims = (e.id for e in events)

        junk = Identity(
            identifier_key="vehicle",
            class_name="car",
            plate=JUNK,
            plate_source=PLATE_SOURCE_MINT,
            first_seen=TS,
            last_seen=TS,
        )
        neighbour = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Bo Truck",
            first_seen=TS,
            last_seen=TS,
        )
        face = Identity(
            identifier_key="face",
            class_name="person",
            label="Dana",
            first_seen=TS,
            last_seen=TS,
        )
        session.add_all([junk, neighbour, face])
        session.flush()

        session.add(
            EventIdentity(
                event_id=one_claim,
                identity_id=junk.id,
                identifier_key="vehicle",
                similarity=0.9,
            )
        )
        session.add_all(
            [
                EventIdentity(
                    event_id=two_claims,
                    identity_id=junk.id,
                    identifier_key="vehicle",
                    similarity=0.9,
                ),
                EventIdentity(
                    event_id=two_claims,
                    identity_id=neighbour.id,
                    identifier_key="vehicle",
                    similarity=0.8,
                ),
            ]
        )

        def _read(event_id, **kw):
            row = PlateRead(
                event_id=event_id,
                camera_id="cam1",
                class_name="car",
                identifier_key="vehicle",
                at=TS,
                accepted=True,
                **kw,
            )
            session.add(row)
            session.flush()
            return row.id

        reads = SimpleNamespace(
            # The two live repairs: one confirmed after a correction, one
            # confirmed as read.
            corrected=_read(
                one_claim, raw_text="TYB5O6", text="TYB5O6",
                corrected_text=TRUTH, verdict="wrong",
            ),
            confirmed=_read(
                one_claim, raw_text=TRUTH, text=TRUTH, verdict="confirmed"
            ),
            unjudged=_read(one_claim, raw_text="ZZZ111", text="ZZZ111"),
            ambiguous=_read(
                two_claims, raw_text="AAA111", text="AAA111", verdict="confirmed"
            ),
            orphan=_read(
                no_claims, raw_text="BBB222", text="BBB222", verdict="confirmed"
            ),
        )
        session.commit()
        ids = SimpleNamespace(
            junk=junk.id,
            neighbour=neighbour.id,
            face=face.id,
            one_claim=one_claim,
            two_claims=two_claims,
            no_claims=no_claims,
            reads=reads,
        )

    return SimpleNamespace(
        client=TestClient(create_app(config)),
        Session=Session,
        config=config,
        ids=ids,
    )


def _post_plate(env, identity_id, **data):
    return env.client.post(
        f"/identities/{identity_id}/plate", data=data, follow_redirects=False
    )


def _identity(env, identity_id) -> Identity:
    with env.Session() as session:
        return session.get(Identity, identity_id)


def _resolver(env) -> IdentityResolver:
    """A resolver on the store the app already opened — embedded Qdrant is
    one client per path per machine."""
    return IdentityResolver(
        env.config.identity, get_shared_store(env.config.identity.vector_db_path)
    )


def _resolve(env, session, *, plate, vector=VEHICLE_VECTOR, resolver=None):
    return (resolver or _resolver(env)).resolve(
        session,
        identifier_key="vehicle",
        class_name="car",
        vector=vector.tolist(),
        plate=plate,
        timestamp=TS,
        threshold=0.8,
    )


# -- 1-3. clearing, and making the clear stick -----------------------------


def test_a_plate_read_at_mint_can_be_cleared(plate_env):
    """AC 1. Nothing in the console could reach this plate: reverting a
    learned plate needs an `EventIdentity.learned_plate` row to revert,
    and a plate taken at mint has none. The identity carried `111111`
    with no way to say otherwise."""
    with plate_env.Session() as session:
        assert not session.scalars(
            select(EventIdentity).where(EventIdentity.learned_plate)
        ).all()  # no learned link exists, so no revert path exists

    r = _post_plate(plate_env, plate_env.ids.junk, plate="")

    assert r.status_code == 303
    assert r.headers["location"] == f"/identities/{plate_env.ids.junk}"
    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate is None
    assert identity.plate_source == PLATE_SOURCE_OPERATOR


def test_the_clear_survives_the_next_sighting(plate_env):
    """**The heart of the issue.** The resolver learns a plate when the
    identity has none — so clearing one hands the next sighting exactly
    the condition it learns on, and without the `operator` guard the junk
    string comes straight back and the correction undoes itself with no
    trace.

    A clear that does not stick is not a fix, so this test must fail
    loudly if the guard is ever dropped.
    """
    _post_plate(plate_env, plate_env.ids.junk, plate="")
    vectors = get_shared_store(plate_env.config.identity.vector_db_path)
    vectors.add("vehicle", VEHICLE_VECTOR, plate_env.ids.junk)

    with plate_env.Session() as session:
        # The same vehicle comes past, and OCR reads the same junk again.
        res = _resolve(plate_env, session, plate=JUNK)
        session.commit()

    # It is still recognised — the clear withdraws a plate, not a vehicle.
    assert res.identity is not None and res.identity.id == plate_env.ids.junk
    assert res.matched_by == "visual"
    assert res.learned_plate is False
    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate is None  # the correction stood
    assert identity.plate_source == PLATE_SOURCE_OPERATOR


def test_the_same_sighting_would_have_re_learned_without_the_lock(plate_env):
    """The control for the test above, and the reason it is not passing by
    accident: the identical flow, differing only in who emptied the plate.

    A vehicle that simply has no plate yet is *supposed* to learn one from
    a matching sighting (PRD §6.4). So the sighting really does reach the
    learn path, and `plate_source` is the only thing standing between it
    and the junk string coming back.
    """
    with plate_env.Session() as session:
        identity = session.get(Identity, plate_env.ids.junk)
        identity.plate = None
        identity.plate_source = None  # not an operator's doing
        session.commit()
    vectors = get_shared_store(plate_env.config.identity.vector_db_path)
    vectors.add("vehicle", VEHICLE_VECTOR, plate_env.ids.junk)

    with plate_env.Session() as session:
        res = _resolve(plate_env, session, plate=JUNK)
        session.commit()

    assert res.learned_plate is True
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK


def test_relearn_hands_the_vehicle_back_to_automatic_learning(plate_env):
    """The lock is a decision, so it has to be reversible: an operator who
    cleared a plate to stop the guessing can ask for the guessing back."""
    r = _post_plate(plate_env, plate_env.ids.junk, plate="", relearn="1")
    assert r.status_code == 303

    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate is None
    assert identity.plate_source is None  # unknown, not locked

    vectors = get_shared_store(plate_env.config.identity.vector_db_path)
    vectors.add("vehicle", VEHICLE_VECTOR, plate_env.ids.junk)
    with plate_env.Session() as session:
        res = _resolve(plate_env, session, plate="333333")
        session.commit()

    assert res.learned_plate is True
    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate == "333333"
    assert identity.plate_source == PLATE_SOURCE_LEARNED


# -- 4-7. editing --------------------------------------------------------


def test_an_edit_overwrites_and_normalizes_what_was_typed(plate_env):
    """`Identity.plate` is compared to `normalize_plate`'d OCR output, so
    a plate stored the way it was typed would never match a future read.
    The edit would look like it worked and do nothing."""
    r = _post_plate(plate_env, plate_env.ids.junk, plate="tyb-506")

    assert r.status_code == 303
    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate == TRUTH  # not "tyb-506", not "TYB-506"
    assert identity.plate_source == PLATE_SOURCE_OPERATOR


@pytest.mark.parametrize("typed", ["---", "!?", "()"])
def test_an_edit_refuses_input_that_normalizes_to_nothing(plate_env, typed):
    """Characters that all fall out of `normalize_plate` would store an
    empty string, which reads as "no plate" while looking like a
    successful edit — the same refusal `plate_correct` already makes."""
    r = _post_plate(plate_env, plate_env.ids.junk, plate=typed)

    assert r.status_code == 400
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK  # untouched


def test_whitespace_alone_is_a_clear_not_an_error(plate_env):
    """The boundary next to the refusal above: blank-after-strip means
    "no value" everywhere in this console (`verdict.strip() or None`), and
    an operator emptying the field may well leave a space in it. Refusing
    that would make the clear — the thing this issue exists to provide —
    fail on a keystroke."""
    r = _post_plate(plate_env, plate_env.ids.junk, plate="   ")

    assert r.status_code == 303
    assert _identity(plate_env, plate_env.ids.junk).plate is None


def test_a_face_identity_has_no_plate_to_set_but_may_still_be_cleared(plate_env):
    """Setting is refused where the identifier does not do plates. Clearing
    is always allowed: a plate learned under an older config must stay
    removable, which is the whole complaint."""
    assert _post_plate(plate_env, plate_env.ids.face, plate="ABC123").status_code == 400

    with plate_env.Session() as session:
        session.get(Identity, plate_env.ids.face).plate = "OLD123"
        session.commit()

    assert _post_plate(plate_env, plate_env.ids.face, plate="").status_code == 303
    assert _identity(plate_env, plate_env.ids.face).plate is None


def test_an_edit_refuses_a_plate_another_identity_holds_unless_confirmed(plate_env):
    """Two identities sharing a plate makes the plate-first lookup pick
    between them arbitrarily. The console should not be the thing that
    manufactures that — but an operator who knows about the duplicate (two
    plates genuinely re-issued, a temporary transfer) may still say so."""
    with plate_env.Session() as session:
        session.get(Identity, plate_env.ids.neighbour).plate = TRUTH
        session.commit()

    refused = _post_plate(plate_env, plate_env.ids.junk, plate=TRUTH)
    assert refused.status_code == 409
    assert TRUTH in refused.json()["detail"]
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK

    allowed = _post_plate(plate_env, plate_env.ids.junk, plate=TRUTH, confirm="1")
    assert allowed.status_code == 303
    assert _identity(plate_env, plate_env.ids.junk).plate == TRUTH


# -- 8-10. moving a plate to the vehicle it belongs to ---------------------


def _move(env, identity_id, **data):
    return env.client.post(
        f"/identities/{identity_id}/plate/move", data=data, follow_redirects=False
    )


def test_moving_a_plate_clears_the_source_and_locks_both(plate_env):
    """The plate was on the wrong vehicle, which is two statements: this
    one does not have it, and that one does. Both sides are the operator's
    word, so both are locked — re-learning the same string onto the source
    is the same mistake by another route."""
    r = _move(plate_env, plate_env.ids.junk, target_id=plate_env.ids.neighbour)

    assert r.status_code == 303
    # Attention follows the plate.
    assert r.headers["location"] == f"/identities/{plate_env.ids.neighbour}"
    source = _identity(plate_env, plate_env.ids.junk)
    target = _identity(plate_env, plate_env.ids.neighbour)
    assert source.plate is None
    assert target.plate == JUNK
    assert source.plate_source == PLATE_SOURCE_OPERATOR
    assert target.plate_source == PLATE_SOURCE_OPERATOR


def test_moving_refuses_a_cross_kind_target_and_itself(plate_env):
    """A vehicle plate on a face identity is nonsense the store should not
    hold — the same rule, and the same reason, as the merge endpoint's."""
    assert _move(
        plate_env, plate_env.ids.junk, target_id=plate_env.ids.face
    ).status_code == 400
    assert _move(
        plate_env, plate_env.ids.junk, target_id=plate_env.ids.junk
    ).status_code == 400
    assert _move(plate_env, plate_env.ids.junk, target_id=9999).status_code == 404
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK


def test_moving_onto_an_occupied_plate_names_both_before_overwriting(plate_env):
    """The target already carries a plate, so the move destroys one. The
    refusal names both, because "which plate am I about to lose" is the
    question the operator needs answered."""
    with plate_env.Session() as session:
        session.get(Identity, plate_env.ids.neighbour).plate = TRUTH
        session.commit()

    refused = _move(plate_env, plate_env.ids.junk, target_id=plate_env.ids.neighbour)
    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert JUNK in detail and TRUTH in detail

    allowed = _move(
        plate_env, plate_env.ids.junk, target_id=plate_env.ids.neighbour, confirm="1"
    )
    assert allowed.status_code == 303
    assert _identity(plate_env, plate_env.ids.neighbour).plate == JUNK
    assert _identity(plate_env, plate_env.ids.junk).plate is None


# -- 11. the point of the whole exercise ----------------------------------


def test_the_corrected_plate_wins_the_next_read_outright(plate_env):
    """AC 3. A plate match beats visual similarity (PRD §6.4), so the
    corrected string is not merely stored — it decides where the next
    sighting of that number goes."""
    assert _post_plate(plate_env, plate_env.ids.junk, plate="tyb 506").status_code == 303

    with plate_env.Session() as session:
        # A crop that looks like nothing in any gallery, carrying the plate.
        res = _resolve(plate_env, session, plate=TRUTH, vector=_unit(0.0, 0.0, 1.0, 0.0))
        session.commit()

    assert res.identity is not None
    assert res.identity.id == plate_env.ids.junk
    assert res.matched_by == "plate"
    assert res.similarity == 1.0
    assert not res.is_new  # emphatically not a fresh identity


# -- 12-14. applying a read to the identity that made it ------------------


def _apply(env, read_id, **data):
    return env.client.post(
        f"/plates/{read_id}/apply-identity", data=data, follow_redirects=False
    )


@pytest.mark.parametrize("which", ["corrected", "confirmed"])
def test_applying_a_read_writes_the_value_an_operator_stands_behind(plate_env, which):
    """The two live repairs, both shapes: confirmed after a correction
    (the typed truth wins) and confirmed as read (the OCR text does).
    Either way it is the same write as the identity page's edit — one
    code path, so the two cannot drift."""
    read_id = getattr(plate_env.ids.reads, which)

    r = _apply(plate_env, read_id, back="/plates?status=all")

    assert r.status_code == 303
    assert r.headers["location"] == "/plates?status=all"  # back where they were
    identity = _identity(plate_env, plate_env.ids.junk)
    assert identity.plate == TRUTH
    assert identity.plate_source == PLATE_SOURCE_OPERATOR


def test_applying_declines_when_the_event_cannot_say_whose_plate_it_is(plate_env):
    """Event 30's shape: two active vehicle claims, so applying the read
    would have to guess which vehicle the plate belongs to — the exact
    failure this cluster of issues exists to stop. It names both and
    sends the operator to the identity page instead."""
    refused = _apply(plate_env, plate_env.ids.reads.ambiguous)
    assert refused.status_code == 400
    detail = refused.json()["detail"]
    assert str(plate_env.ids.junk) in detail and str(plate_env.ids.neighbour) in detail

    orphan = _apply(plate_env, plate_env.ids.reads.orphan)
    assert orphan.status_code == 400

    # Neither wrote anything.
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK
    assert _identity(plate_env, plate_env.ids.neighbour).plate is None


def test_applying_declines_a_read_nobody_judged(plate_env):
    """The button applies a value an operator stands behind. An unjudged
    read is the OCR's opinion, which is what put `111111` on an identity
    in the first place."""
    assert _apply(plate_env, plate_env.ids.reads.unjudged).status_code == 400
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK


# -- 15-17. audit, roles, redaction ---------------------------------------


def _user(env, username, role):
    with env.Session() as session:
        session.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=TS,
            )
        )
        session.commit()


def _login(env, username):
    r = env.client.post(
        "/login",
        data={"username": username, "password": "hunter2!", "next": "/"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text[:200]


def test_every_plate_write_is_audited_with_its_actor(plate_env):
    """AC 4. The audit row is what makes the override accountable — it is
    the one write in the identity store a human authored directly."""
    _user(plate_env, "eve", "edit")
    _login(plate_env, "eve")

    # Three distinct writes. The plate differs from the read's, so the
    # move does not park TRUTH on the neighbour and turn the apply into a
    # duplicate refusal — a refused write is correctly *not* audited.
    assert _post_plate(plate_env, plate_env.ids.junk, plate="ZZ9999").status_code == 303
    assert _move(
        plate_env, plate_env.ids.junk, target_id=plate_env.ids.neighbour
    ).status_code == 303
    assert _apply(plate_env, plate_env.ids.reads.confirmed).status_code == 303

    with plate_env.Session() as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.id)).all()
    paths = [row.path for row in rows]
    assert f"/identities/{plate_env.ids.junk}/plate" in paths
    assert f"/identities/{plate_env.ids.junk}/plate/move" in paths
    assert f"/plates/{plate_env.ids.reads.confirmed}/apply-identity" in paths
    # Denormalized on purpose: it must still say who acted after the
    # account is renamed or deleted.
    assert {row.username for row in rows if "plate" in row.path} == {"eve"}


def test_plate_writes_need_the_edit_rung(plate_env):
    """Enforcement lives in the one auth middleware, not on the routes, so
    the floor is asked of `required_role` and then observed live."""
    paths = (
        f"/identities/{plate_env.ids.junk}/plate",
        f"/identities/{plate_env.ids.junk}/plate/move",
        f"/plates/{plate_env.ids.reads.confirmed}/apply-identity",
    )
    for path in paths:
        assert required_role("POST", path) == "edit"

    for role in ("restricted", "view"):
        _user(plate_env, f"{role}-user", role)
        _login(plate_env, f"{role}-user")
        for path in paths:
            assert plate_env.client.post(
                path, data={"plate": TRUTH, "target_id": plate_env.ids.neighbour}
            ).status_code == 403
    assert _identity(plate_env, plate_env.ids.junk).plate == JUNK

    _user(plate_env, "eve", "edit")
    _login(plate_env, "eve")
    assert _post_plate(plate_env, plate_env.ids.junk, plate=TRUTH).status_code == 303


def test_a_plate_is_withheld_below_the_naming_floor(plate_env):
    """The unit half of CLD-111's rule. The screens that render a plate
    are denied to `restricted` wholesale, so the disclosure walk never
    reaches this — `identity_plate()` is defence in depth against a future
    floor move, and the substitution itself is what can be pinned.
    """
    with plate_env.Session() as session:
        identity = session.get(Identity, plate_env.ids.junk)
        assert redaction.plate_for(identity, allowed=True) == JUNK
        assert redaction.plate_for(identity, allowed=False) is None
        assert redaction.plate_for(None, allowed=True) is None

    # The positive half at the view level: a viewer above the floor reads
    # the plate on the identity page.
    _user(plate_env, "vera", "view")
    _login(plate_env, "vera")
    page = plate_env.client.get(f"/identities/{plate_env.ids.junk}").text
    assert JUNK in page
