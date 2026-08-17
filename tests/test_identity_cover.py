"""An identity's cover photo, and who gets to choose it (CLD-137).

`Identity.best_crop_path` was first-write-wins at every writer, which is
how an identity ends up wearing the face of the match that founded it —
including when that match was wrong. Correcting the mistake did not
correct the picture: the event could be unlinked, its vectors stripped
and its plate reverted, and the identity kept showing a crop of somebody
else in every list on the console.

Three things this pins:

* **The cover follows the evidence.** When the crop that supplied it
  stops being this identity's, it is re-derived from what the identity
  still owns — best detection confidence first, verified annotations
  after. NULL is a legitimate answer.
* **An operator can choose, and the choice sticks** against automatic
  recompute — but not against that same operator's later, contradicting
  action. "This event is not this identity" and "this event's crop
  represents this identity" cannot both stand.
* **A cover can outlive its file.** `purge_window` deletes crop files
  while leaving Identity rows, so a cover can point at nothing; the
  screens must render their placeholder rather than a broken image.

Fixtures follow `tests/test_event_identity_edit.py` — stub embedder,
colour-keyed crops, no model weights.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    AuditLog,
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    LibraryItem,
    LibrarySource,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role
from test_event_identity_edit import COLOURS, StubEmbedder

TS = datetime(2026, 8, 11, 9, 0, 0)

#: Crops, and the confidence of the detection that produced each. The
#: ordering is deliberately not the chronological one: `green` is the
#: better crop and `blue` the later, so "best" and "most recent" give
#: different answers.
FOUNDING = "cam1/red.jpg"
BEST_REMAINING = "cam1/green.jpg"
LATER_WORSE = "cam1/blue.jpg"
ANOTHERS = "cam1/lone.jpg"
VERIFIED_ANNOTATION = "cam1/anno.jpg"


@pytest.fixture
def cover_env(tmp_path, monkeypatch):
    """An identity founded by event A, still claiming event B.

    Its cover is A's crop — the founding match's picture. B carries two
    detections so the re-pick has something to rank, and a second
    identity owns a crop of its own so ownership can be refused.
    """
    monkeypatch.setattr(
        "siteloom.identity.embedders.build_embedder",
        lambda algo, device="mps", projection_path=None: StubEmbedder(),
    )
    media = tmp_path / "media"
    (media / "cam1").mkdir(parents=True)
    for name, path in (
        ("red", FOUNDING),
        ("green", BEST_REMAINING),
        ("blue", LATER_WORSE),
        ("red", ANOTHERS),
        ("green", VERIFIED_ANNOTATION),
        ("blue", "cam1/junk.jpg"),
        ("blue", "cam1/bad.jpg"),
    ):
        square = np.full((16, 16, 3), COLOURS[name][0], dtype=np.uint8)
        cv2.imwrite(str(media / path), square)

    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/cover.db", media_dir=str(media)
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    with Session() as session:
        session.add(Camera(id="cam1", site_id="t", name="Cam One"))
        subject = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Aleks Corolla",
            first_seen=TS,
            last_seen=TS,
            appearance_count=4,
            best_crop_path=FOUNDING,  # the founding match's picture
        )
        other = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Bo Truck",
            first_seen=TS,
            last_seen=TS,
            appearance_count=1,
            best_crop_path=ANOTHERS,
        )
        session.add_all([subject, other])
        session.flush()

        def _event(track, claims, crops):
            event = Event(
                camera_id="cam1",
                track_id=track,
                class_name="car",
                first_seen=TS,
                last_seen=TS,
                detection_count=len(crops),
                best_crop_path=crops[0][0],
                best_confidence=crops[0][1],
            )
            session.add(event)
            session.flush()
            for i, (crop, confidence) in enumerate(crops):
                session.add(
                    Detection(
                        event_id=event.id,
                        timestamp=datetime(2026, 8, 11, 9, 0, i * 10),
                        class_name="car",
                        confidence=confidence,
                        bbox="[1, 2, 3, 4]",
                        crop_path=crop,
                    )
                )
            link = EventIdentity(
                event_id=event.id,
                identity_id=claims,
                identifier_key="vehicle",
                similarity=0.9,
                matched_by="visual",
            )
            session.add(link)
            session.flush()
            return event.id, link.id

        # A founded the identity; B is the visit it still has, and its
        # better crop is the earlier of its two detections.
        event_a, link_a = _event(1, subject.id, [(FOUNDING, 0.95)])
        event_b, link_b = _event(
            2, subject.id, [(BEST_REMAINING, 0.80), (LATER_WORSE, 0.60)]
        )
        event_c, link_c = _event(3, other.id, [(ANOTHERS, 0.90)])

        source = LibrarySource(path=str(tmp_path / "library"), added_at=TS)
        session.add(source)
        session.flush()
        item = LibraryItem(
            source_id=source.id,
            path=str(tmp_path / "library" / "photo.jpg"),
            kind="image",
            mtime=TS,
        )
        session.add(item)
        session.flush()
        annotations = {}
        # The two that must never be chosen are created *first*, so they
        # sort ahead of the verified one: a fallback that forgot to filter
        # would reach for one of them rather than quietly agreeing by
        # accident of ordering.
        for name, crop, verified, rejected in (
            ("unverified", "cam1/junk.jpg", False, False),
            ("rejected", "cam1/bad.jpg", True, True),
            ("verified", VERIFIED_ANNOTATION, True, False),
        ):
            annotation = Annotation(
                item_id=item.id,
                bbox="[0.1, 0.1, 0.5, 0.5]",
                class_name="car",
                identity_id=subject.id,
                crop_path=crop,
                verified=verified,
                verified_by="human" if verified else None,
                rejected=rejected,
                created_at=TS,
            )
            session.add(annotation)
            session.flush()
            annotations[name] = annotation.id

        session.commit()
        ids = SimpleNamespace(
            subject=subject.id,
            other=other.id,
            event_a=event_a,
            event_b=event_b,
            event_c=event_c,
            link_a=link_a,
            link_b=link_b,
            link_c=link_c,
            annotations=annotations,
        )

    return SimpleNamespace(
        client=TestClient(create_app(config)),
        Session=Session,
        config=config,
        media=media,
        ids=ids,
    )


def _cover(env, identity_id=None):
    with env.Session() as session:
        return session.get(Identity, identity_id or env.ids.subject)


def _unlink(env, link_id, event_id=None):
    return env.client.post(
        f"/events/{event_id or env.ids.event_a}/identity/{link_id}/unlink",
        follow_redirects=False,
    )


def _set_cover(env, crop_path, identity_id=None):
    return env.client.post(
        f"/identities/{identity_id or env.ids.subject}/cover",
        data={"crop_path": crop_path},
        follow_redirects=False,
    )


# -- 1-4. the cover follows the evidence ----------------------------------


def test_unlinking_the_founding_event_repicks_the_cover(cover_env):
    """The headline. Correcting a wrong founding match used to strip its
    vectors and revert its plate while leaving the identity wearing its
    picture — the one thing an operator sees in every list."""
    assert _cover(cover_env).best_crop_path == FOUNDING

    r = _unlink(cover_env, cover_env.ids.link_a)
    assert r.status_code == 303

    assert _cover(cover_env).best_crop_path == BEST_REMAINING


def test_the_repick_takes_the_best_crop_not_the_most_recent(cover_env):
    """Ranked the way `Event.best_crop_path` is ranked within an event,
    applied across the identity: detector confidence, not recency. The
    later detection here is the worse one, so the two rules disagree."""
    _unlink(cover_env, cover_env.ids.link_a)

    identity = _cover(cover_env)
    assert identity.best_crop_path == BEST_REMAINING  # 0.80, earlier
    assert identity.best_crop_path != LATER_WORSE  # 0.60, later


def test_unlinking_an_unrelated_event_leaves_the_cover_alone(cover_env):
    """No churn: the crop that supplies the cover is still this
    identity's, so nothing about it has changed."""
    r = _unlink(cover_env, cover_env.ids.link_b, event_id=cover_env.ids.event_b)
    assert r.status_code == 303

    assert _cover(cover_env).best_crop_path == FOUNDING


def test_losing_every_claim_leaves_no_cover_rather_than_a_stale_one(cover_env):
    """NULL is a legitimate outcome — every template already guards on
    the field, so it renders the placeholder. Keeping the last wrong crop
    because there is nothing to replace it with would be the bug."""
    with cover_env.Session() as session:  # remove the annotation fallback too
        for annotation_id in cover_env.ids.annotations.values():
            session.get(Annotation, annotation_id).identity_id = None
        session.commit()

    _unlink(cover_env, cover_env.ids.link_a)
    _unlink(cover_env, cover_env.ids.link_b, event_id=cover_env.ids.event_b)

    assert _cover(cover_env).best_crop_path is None
    page = cover_env.client.get(f"/identities/{cover_env.ids.subject}")
    assert page.status_code == 200
    assert f"/media/{FOUNDING}" not in page.text


def test_a_verified_annotation_is_the_fallback_and_a_guess_is_not(cover_env):
    """With no live claim left, the library is what the identity still
    owns — but only its verified, non-rejected annotations. An unverified
    auto annotation is a guess, and the rule that a guess is not training
    data applies at least as hard to the picture that names someone."""
    _unlink(cover_env, cover_env.ids.link_a)
    _unlink(cover_env, cover_env.ids.link_b, event_id=cover_env.ids.event_b)

    identity = _cover(cover_env)
    assert identity.best_crop_path == VERIFIED_ANNOTATION
    assert identity.best_crop_path not in ("cam1/junk.jpg", "cam1/bad.jpg")


# -- 6-7. the other editing paths -----------------------------------------


def test_reassigning_repicks_the_old_cover_and_leaves_the_new_one(cover_env):
    """Reassign duplicates unlink's steps inline rather than calling it,
    so it needs the recompute of its own — and it must not touch the
    target's cover, which nothing about this correction changed."""
    r = cover_env.client.post(
        f"/events/{cover_env.ids.event_a}/identity/{cover_env.ids.link_a}/reassign",
        data={"identity_id": cover_env.ids.other},
        follow_redirects=False,
    )
    assert r.status_code == 303

    assert _cover(cover_env).best_crop_path == BEST_REMAINING
    assert _cover(cover_env, cover_env.ids.other).best_crop_path == ANOTHERS


def test_a_wrong_verdict_changes_no_cover(cover_env):
    """Judging is not editing — the endpoint's own contract. The claim
    still stands after a `wrong` verdict, so the crop is still this
    identity's; unlink is the route that says otherwise."""
    r = cover_env.client.post(
        f"/events/{cover_env.ids.event_a}/identity/{cover_env.ids.link_a}/verdict",
        data={"verdict": "wrong"},
    )
    assert r.status_code == 200

    assert _cover(cover_env).best_crop_path == FOUNDING


# -- 8-11. the operator's choice ------------------------------------------


def test_choosing_a_cover_locks_it_against_recompute(cover_env):
    """The lock is what makes the choice worth making: without it the
    next unrelated correction would quietly overwrite what the operator
    picked, and they would have no way to tell it had happened."""
    r = _set_cover(cover_env, LATER_WORSE)
    assert r.status_code == 303
    assert r.headers["location"] == f"/identities/{cover_env.ids.subject}"

    identity = _cover(cover_env)
    assert identity.best_crop_path == LATER_WORSE  # not the best-ranked one
    assert identity.cover_locked is True

    # An unrelated correction leaves it alone.
    _unlink(cover_env, cover_env.ids.link_a)
    identity = _cover(cover_env)
    assert identity.best_crop_path == LATER_WORSE
    assert identity.cover_locked is True


def test_unlinking_the_event_behind_a_locked_cover_wins(cover_env):
    """The lock protects an operator's choice from *automatic* recompute,
    not from the same operator's later contradiction. "This event is not
    this identity" and "this event's crop represents this identity"
    cannot both stand, and the later statement is the live one."""
    _set_cover(cover_env, BEST_REMAINING)  # a crop event B supplied

    _unlink(cover_env, cover_env.ids.link_b, event_id=cover_env.ids.event_b)

    identity = _cover(cover_env)
    assert identity.cover_locked is False  # the lock went with the claim
    assert identity.best_crop_path == FOUNDING  # re-picked from what is left


def test_an_empty_choice_hands_the_cover_back_to_the_system(cover_env):
    """The undo, deliberately the same shape as the plate `relearn`: an
    operator who made a bad choice does not have to hunt for the crop the
    system would have picked."""
    _set_cover(cover_env, LATER_WORSE)

    r = _set_cover(cover_env, "")
    assert r.status_code == 303

    identity = _cover(cover_env)
    assert identity.cover_locked is False
    assert identity.best_crop_path == FOUNDING  # the automatic pick, 0.95


@pytest.mark.parametrize(
    "crop_path",
    [
        ANOTHERS,  # well-formed, and someone else's
        "../../etc/passwd",  # traversal
        "cam1/never-existed.jpg",  # nothing owns it
    ],
)
def test_choosing_a_crop_this_identity_does_not_own_is_refused(cover_env, crop_path):
    """`best_crop_path` is rendered as `/media/{path}` on a view-level
    screen, so an unchecked form field would let an operator point one
    identity's cover at anything the media route will serve."""
    r = _set_cover(cover_env, crop_path)

    assert r.status_code == 400
    identity = _cover(cover_env)
    assert identity.best_crop_path == FOUNDING
    assert identity.cover_locked is False


# -- 12-13. surgery carries the lock --------------------------------------


def test_merge_keeps_the_targets_cover_and_adopts_a_lock_with_an_empty_one(cover_env):
    """The target keeps its own cover — that is the stated behaviour. But
    when it has none and adopts the source's, it must adopt the lock too,
    or the merge silently downgrades an operator's choice to automatic
    and the next recompute clobbers it."""
    _set_cover(cover_env, LATER_WORSE)  # the source's cover, chosen and locked

    # Target already has a cover: it keeps it, unlocked as it was.
    r = cover_env.client.post(
        f"/identities/{cover_env.ids.subject}/merge",
        data={"target_id": cover_env.ids.other},
        follow_redirects=False,
    )
    assert r.status_code == 303
    target = _cover(cover_env, cover_env.ids.other)
    assert target.best_crop_path == ANOTHERS
    assert target.cover_locked is False


def test_merge_into_an_uncovered_target_carries_the_lock(cover_env):
    _set_cover(cover_env, LATER_WORSE)
    with cover_env.Session() as session:
        session.get(Identity, cover_env.ids.other).best_crop_path = None
        session.commit()

    cover_env.client.post(
        f"/identities/{cover_env.ids.subject}/merge",
        data={"target_id": cover_env.ids.other},
        follow_redirects=False,
    )

    target = _cover(cover_env, cover_env.ids.other)
    assert target.best_crop_path == LATER_WORSE
    assert target.cover_locked is True  # the operator's choice survived


def test_splitting_away_a_locked_cover_clears_the_lock(cover_env):
    """Same rule as the unlink: the crop left with the annotations, so
    the statement it stood for no longer holds for this identity.

    *Which* crop the source lands on is not asserted here. Split has its
    own annotation-scoped re-pick, which this issue deliberately leaves
    alone (§2) — it answers a narrower question and has its own tests, so
    pinning its choice from here would be this file claiming ownership of
    a rule it does not define.
    """
    _set_cover(cover_env, VERIFIED_ANNOTATION)

    r = cover_env.client.post(
        f"/identities/{cover_env.ids.subject}/split",
        data={"annotation_ids": str(cover_env.ids.annotations["verified"])},
        follow_redirects=False,
    )
    assert r.status_code == 303

    source = _cover(cover_env)
    assert source.cover_locked is False  # the lock left with the crop
    assert source.best_crop_path != VERIFIED_ANNOTATION  # re-picked
    assert source.best_crop_path is not None  # and not merely emptied


# -- 14. a cover can outlive its file -------------------------------------


def test_a_cover_whose_file_is_gone_renders_the_placeholder(cover_env):
    """`purge_window` deletes crop files for the events it removes while
    leaving Identity rows standing, so an identity can keep a path to a
    file that is gone. Today that is a broken-image icon on the busiest
    screens in the console."""
    listing_before = cover_env.client.get("/identities").text
    assert f"/media/{FOUNDING}" in listing_before  # the control

    (cover_env.media / FOUNDING).unlink()

    # The media route itself 404s it — which is the broken image.
    assert cover_env.client.get(f"/media/{FOUNDING}").status_code == 404
    # ... so neither screen may offer it as an <img>.
    assert f"/media/{FOUNDING}" not in cover_env.client.get("/identities").text
    detail = cover_env.client.get(f"/identities/{cover_env.ids.subject}")
    assert detail.status_code == 200
    assert f'src="/media/{FOUNDING}"' not in detail.text


# -- 15. roles and audit ---------------------------------------------------


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


def test_choosing_a_cover_needs_the_edit_rung_and_is_audited(cover_env):
    """Enforcement and auditing live in the one middleware, so this asks
    the floor and then observes it, rather than re-implementing either."""
    path = f"/identities/{cover_env.ids.subject}/cover"
    assert required_role("POST", path) == "edit"

    for role in ("restricted", "view"):
        _user(cover_env, f"{role}-user", role)
        _login(cover_env, f"{role}-user")
        assert cover_env.client.post(
            path, data={"crop_path": LATER_WORSE}
        ).status_code == 403
    assert _cover(cover_env).best_crop_path == FOUNDING

    _user(cover_env, "eve", "edit")
    _login(cover_env, "eve")
    assert _set_cover(cover_env, LATER_WORSE).status_code == 303

    with cover_env.Session() as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.id)).all()
    covers = [row for row in rows if row.path == path]
    assert len(covers) == 1  # the refused ones are not audited
    assert covers[0].username == "eve"
