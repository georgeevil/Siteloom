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

from siteloom.store import Detection, Event, Identity

log = logging.getLogger(__name__)

def shared_store(config, action: str):
    """The process-wide vector store, or an actionable 503.

    Embedded Qdrant allows ONE client per path per machine (see
    identity/vectors.py), so a running backfill/index/frigate job blocks
    every edit here. Saying which job and what to do about it is the
    difference between a bug report and a wait.
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
    return removed
