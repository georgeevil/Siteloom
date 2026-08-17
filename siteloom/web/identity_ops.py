"""Operator-driven identity surgery, shared by the event and identity routes.

Every one of these actions — merge, split, link, unlink, reassign — is a
human overruling the resolver, and they all face the same two facts.

First, the DB and the vector store have to move together. A claim
corrected in the database while its embedding stays in the wrong
gallery is not a correction at all: the same wrong match returns on the
next frame, and the operator's work looks undone. That is why these
helpers refuse (503) rather than proceed when another process holds the
embedded store — half a correction is worse than none.

Second, a name the system cannot see is not a name (identity/enroll.py).
So attaching an identity to an event offers to enroll the event's own
crop under it — the crop live matching already embedded, which is what
keeps the manual link in the same vector space as everything else ("one
crop, two jobs").
"""

from __future__ import annotations

import logging

from fastapi import HTTPException
from sqlalchemy import select

from siteloom.identity.plates import normalize_plate
from siteloom.store import Detection, Event, Identity
from siteloom.store.models import PLATE_SOURCE_OPERATOR

log = logging.getLogger(__name__)

#: How many cover candidates the identity page offers. A strip an
#: operator scans, not the identity's whole history.
COVER_CANDIDATES = 24

#: The bound `owns_crop` checks against — every crop the identity could
#: legitimately wear, rather than the strip the page happened to render.
COVER_CANDIDATES_MAX = 10_000

def shared_store(config, action: str):
    """The process-wide vector store, or an actionable 503.

    Embedded Qdrant allows ONE client per path per machine (see
    identity/vectors.py), so a running backfill/index/frigate job blocks
    every edit here. Saying which job and what to do about it is the
    difference between a bug report and a wait.

    Every operator-facing path that needs the store goes through this
    one function — identity surgery, confirming a face proposal,
    assigning a custom class, the enrolment sweep, the import wizard's
    indexer (CLD-62). They are all the same situation: an operator
    working the console while ingest runs is the *normal* case, not an
    exotic failure, and a second spelling of this refusal is a second
    chance to phrase it as a 500. (`web/recognition_api.py` deliberately
    does not use this helper: its callers are CompreFace clients, not
    operators, so its reads degrade to a marked no-match and its writes
    refuse in CompreFace's own error shape — CLD-110.)

    Resolve it in the request that needs it, never inside a worker
    thread: a 503 the operator can read beats a job that starts, dies out
    of sight, and leaves them watching /jobs for a run that never
    appears.
    """
    from siteloom.identity import get_shared_store

    try:
        return get_shared_store(config.identity.vector_db_path)
    except RuntimeError as exc:
        raise HTTPException(
            503,
            "the vector store is locked by another process — a backfill, "
            "library index, or frigate job is likely running against the "
            f"same database. Wait for it to finish (see /jobs), then retry "
            f"the {action}. ({exc})",
        )


def identifier_embedder(config, identifier_key: str, cache: dict):
    """The embedder an identifier's vectors are built with.

    Re-embedding a stored crop must use the embedder live matching used,
    or the new vector lands in a different space and never matches.
    Building one is expensive (model load), so the app holds a cache
    keyed by algorithm — not by identifier: two identifiers on the same
    algorithm share an embedder, which is what keeps them in one vector
    space. The cache belongs to the app rather than this module so it
    cannot outlive the config it was built from.
    """
    from siteloom.identity import embedders

    ident = config.identity.identifiers.get(identifier_key)
    # Auto-added classes have no configured identifier; they are generic
    # by construction (identity/registry.py).
    algo = ident.algo if ident else "generic"
    if algo not in cache:
        cache[algo] = embedders.build_embedder(
            algo,
            device=config.detection.device,
            projection_path=config.identity.face_projection_path or None,
        )
    return cache[algo]


def max_vectors_for(config, identifier_key: str) -> int:
    ident = config.identity.identifiers.get(identifier_key)
    return ident.max_vectors_per_identity if ident else 20


def event_crop_paths(session, event: Event) -> list[str]:
    """Every crop file this event produced.

    These are the provenance handles for the vectors the event
    contributed to an identity (CLD-84): live matching embeds a
    detection crop and records its path, so "the vectors this event
    taught that identity" is exactly "the vectors made from these
    files". Includes the event thumbnail, which is one of the detection
    crops but is the one a manual enrollment uses.
    """
    paths = [
        p
        for (p,) in session.execute(
            select(Detection.crop_path).where(
                Detection.event_id == event.id, Detection.crop_path.is_not(None)
            )
        )
    ]
    if event.best_crop_path and event.best_crop_path not in paths:
        paths.append(event.best_crop_path)
    return paths


def refresh_vector_count(session, vectors, identity: Identity) -> None:
    """Re-read an identity's vector count from the store.

    `Identity.vector_count` is incremented by several writers and
    drifts; after surgery is exactly when an operator is looking at it.
    """
    identity.vector_count = vectors.count_identity(
        identity.identifier_key, identity.id
    )


def cover_candidates(session, identity: Identity, *, limit: int = 24) -> list[str]:
    """Crops that could represent this identity, best first.

    Detection crops from its *active* links, highest detector confidence
    first — the same ranking `Event.best_crop_path` uses per event,
    applied across the identity — then the crops of its verified,
    non-rejected annotations. Only verified ones: an unverified auto
    annotation is a guess, and the rule that a guess is not training data
    (training/dataset.py) applies at least as hard to the picture that
    names someone in every list on the console.
    """
    from siteloom.store import Annotation, EventIdentity

    paths = [
        p
        for (p,) in session.execute(
            select(Detection.crop_path)
            .join(EventIdentity, EventIdentity.event_id == Detection.event_id)
            .where(
                EventIdentity.identity_id == identity.id,
                EventIdentity.unlinked_at.is_(None),
                Detection.crop_path.is_not(None),
            )
            .order_by(Detection.confidence.desc())
            .limit(limit)
        )
    ]
    if len(paths) < limit:
        paths += [
            p
            for (p,) in session.execute(
                select(Annotation.crop_path)
                .where(
                    Annotation.identity_id == identity.id,
                    Annotation.verified.is_(True),
                    Annotation.rejected.is_(False),
                    Annotation.crop_path.is_not(None),
                )
                .order_by(Annotation.id)
                .limit(limit)
            )
        ]
    # De-duplicate preserving order: one event contributes several crops,
    # and its own best_crop_path is one of them.
    return list(dict.fromkeys(paths))[:limit]


def owns_crop(session, identity: Identity, crop_path: str) -> bool:
    """Whether this crop is one this identity may wear.

    The ownership guard for the operator endpoint — the same shape
    `split_identity` applies to annotation ids ("this endpoint only
    claims to split the identity in the URL"), and here it is also a
    containment concern: `best_crop_path` renders as `/media/{path}` on
    a `view`-level screen, so an unchecked form field would let an
    operator point one identity's cover at anything the media route will
    serve.

    Asked over the identity's whole candidate set, not the strip the
    page happened to show: the limit is presentation, and a form built
    from a wider page must not be refused for it.
    """
    return crop_path in set(cover_candidates(session, identity, limit=COVER_CANDIDATES_MAX))


def recompute_cover(session, identity: Identity, *, dropped) -> bool:
    """Re-derive the cover when the crop that supplied it stops being
    this identity's.

    `dropped` is the set of crops just taken away; None means recompute
    unconditionally — the operator asking for "automatic" again. Returns
    whether the cover changed.
    """
    before = identity.best_crop_path
    if dropped is not None and identity.best_crop_path not in set(dropped):
        # Unlinking an unrelated event must not churn a cover that is
        # still valid. One line, and it covers the no-cover case too:
        # None is never in `dropped`.
        return False
    if identity.cover_locked:
        # The lock protects an operator's choice from *automatic*
        # recompute; it does not protect it from the operator's own
        # later, contradicting action. "This event is not this identity"
        # and "this event's crop represents this identity" cannot both
        # stand, and the later statement wins.
        identity.cover_locked = False
    identity.best_crop_path = next(
        iter(cover_candidates(session, identity, limit=1)), None
    )
    return identity.best_crop_path != before


def enroll_event_crop(
    config, vectors, event: Event, identity: Identity, embedder_cache: dict
) -> bool:
    """Embed this event's best crop into an identity's gallery.

    What makes a manual link visible to future matching rather than a
    note in the database. Returns False (never raises) when there is no
    crop, the gallery is full, or the crop cannot be embedded — a face
    embedder handed a wide vehicle crop legitimately finds nothing, and
    that is a link without enrollment, not a failed request.
    """
    from siteloom.identity.enroll import embed_crop_file
    from siteloom.identity.vectors import SOURCE_MANUAL

    if not event.best_crop_path:
        return False
    if identity.vector_count >= max_vectors_for(config, identity.identifier_key):
        return False
    embedding = embed_crop_file(
        identifier_embedder(config, identity.identifier_key, embedder_cache),
        event.best_crop_path,
    )
    if embedding is None:
        log.info(
            "manual link on event %s: crop %s produced no embedding for %s",
            event.id,
            event.best_crop_path,
            identity.identifier_key,
        )
        return False
    vectors.add(
        identity.identifier_key,
        embedding,
        identity.id,
        source=SOURCE_MANUAL,
        crop_path=event.best_crop_path,
    )
    return True


def revert_learned_plate(session, link, event_id: int) -> None:
    """Undo a plate this very match taught its identity.

    Plate matches win outright (PRD §6.4), so a plate learned from a
    claim the operator has now repudiated would poison every future
    sighting of that number. Scoped to exactly the evidence being
    withdrawn — this is correction, not learning.
    """
    if not link.learned_plate or link.identity_id is None:
        return
    identity = session.get(Identity, link.identity_id)
    if identity is not None and identity.plate:
        log.info(
            "reverting plate %s learned on event %s from identity %s",
            identity.plate,
            event_id,
            identity.id,
        )
        identity.plate = None


def set_identity_plate(
    session, identity, plate: str | None, *, confirm: bool = False
) -> str | None:
    """Write, overwrite or clear an identity's plate on operator authority.

    The one path that may overwrite a non-empty `Identity.plate`: the
    resolver is write-once by design (a plate match beats visual
    similarity outright, PRD §6.4, so a plate that changed itself would
    move every future sighting of that number), and an operator saying
    what the plate is outranks any OCR read.

    Input is normalized, never stored verbatim. `Identity.plate` is
    compared against `normalize_plate`'d OCR output, so a typed
    "TYB-506" kept as typed would never match a future read — the edit
    would look like it worked and quietly do nothing.

    Returns the normalized plate, or None for a clear. Raises 400 on
    input that normalizes to nothing (the rule `plate_correct` already
    applies to a correction) and 409 when another identity of the same
    identifier already carries the plate, unless `confirm`: two
    identities sharing one plate makes the plate-first lookup pick
    between them arbitrarily, and the console should not be the thing
    that manufactures that.

    Rows only — plates are never embedded, so nothing here touches the
    vector store.
    """
    if not plate or not plate.strip():
        identity.plate = None
        identity.plate_source = PLATE_SOURCE_OPERATOR
        return None
    normalized = normalize_plate(plate)
    if not normalized:
        raise HTTPException(400, f"{plate!r} normalizes to nothing")
    if not confirm:
        clash = session.scalar(
            select(Identity).where(
                Identity.identifier_key == identity.identifier_key,
                Identity.plate == normalized,
                Identity.id != identity.id,
            )
        )
        if clash is not None:
            raise HTTPException(
                409,
                f"{normalized} is already on {clash.display_name} "
                f"(identity {clash.id}) — confirm to give it to both",
            )
    identity.plate = normalized
    identity.plate_source = PLATE_SOURCE_OPERATOR
    return normalized


def unlink_claim(session, vectors, event, link, *, when) -> int:
    """Detach one identity claim from an event, and undo what it taught.

    Extracted so the single-claim route and the bulk action (CLD-103)
    cannot drift. The row surviving with its identity, similarity and
    matched_by intact is the record of what the system got wrong;
    negatives are data. What must not survive is the claim's *effect* on
    matching — a gallery polluted by this event keeps re-attracting the
    same wrong match, so the vectors this event contributed are removed
    and any plate it taught is reverted.

    Returns the number of vectors removed, for the caller's log.
    """
    from siteloom.store import Identity

    identity = session.get(Identity, link.identity_id)
    if identity is None:
        link.unlinked_at = when
        return 0
    removed = vectors.delete_by_crops(
        identity.identifier_key,
        identity.id,
        event_crop_paths(session, event),
    )
    link.unlinked_at = when
    # Unlink is the operator asserting the pairing never held, which is a
    # stronger statement than a "wrong" verdict — but it implies one, and
    # /stats counts verdicts.
    link.verdict = "wrong"
    link.verdict_at = when
    revert_learned_plate(session, link, event.id)
    identity.appearance_count = max(identity.appearance_count - 1, 0)
    refresh_vector_count(session, vectors, identity)
    # After the unlink is flushed, so the candidate query cannot see the
    # link it just detached. Autoflush would do it, but a helper whose
    # correctness depends on autoflush ordering is one refactor away from
    # being wrong. The bulk unlink (CLD-103) gets this for free, which is
    # why the function was extracted in the first place.
    session.flush()
    recompute_cover(session, identity, dropped=event_crop_paths(session, event))
    return removed
