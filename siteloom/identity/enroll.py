"""Enrollment: verified face annotations -> the live face collection.

Verifying a proposal in the review UI establishes ground truth; this
step is what makes that truth *operational* — the crop's embedding is
added to the vector store under the labeled Identity, so the person is
recognized on live cameras, by the Frigate consumer, and through the
CompreFace-compatible API. A label without vectors is a name the system
knows but cannot see.

Runs in two places:
- inline when a proposal is confirmed in the review UI (one crop), and
- as `siteloom train enroll`, a sweep over every verified-but-unenrolled
  annotation (the backlog case, idempotent via Annotation.enrolled).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.store import Annotation, Identity

log = logging.getLogger(__name__)


@dataclass
class EnrollStats:
    enrolled: int = 0
    skipped_cap: int = 0
    skipped_no_face: int = 0
    identities: int = 0


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def identity_for_label(session: Session, name: str) -> Identity:
    identity = session.scalar(
        select(Identity).filter_by(identifier_key="face", label=name)
    )
    if identity is None:
        identity = Identity(
            identifier_key="face",
            class_name="person",
            label=name,
            first_seen=_now(),
            last_seen=_now(),
        )
        session.add(identity)
        session.flush()
    return identity


def embed_crop_file(embedder, crop_path: str | None):
    """Re-embed a stored crop JPEG with the identifier's own embedder.

    The stored crop is the SAME image live matching embedded ("one crop,
    two jobs"), so re-embedding it lands in the same vector space. Face
    embedders get a tight-crop fallback: a crop with no detectable face
    inside it is embedded directly rather than losing the sample.
    Returns None for a missing or hopeless crop.
    """
    if not crop_path:
        return None
    import cv2

    crop = cv2.imread(crop_path)
    if crop is None:
        return None
    embedding = embedder.embed(crop)
    if embedding is None and hasattr(embedder, "_recognizer"):
        try:
            resized = cv2.resize(crop, (112, 112))
            feature = (
                embedder._recognizer.feature(resized).flatten().astype("float32")
            )
            embedding = embedder._finish(feature)
        except (cv2.error, AttributeError):
            embedding = None
    return embedding


def embed_annotations(embedder, annotations) -> list[tuple[Annotation, "object"]]:
    """Re-embed the annotations that are allowed to become vectors.

    Only verified, non-rejected annotations with a crop qualify — the
    same rule `enroll_annotation` and `enroll_verified` enforce. An
    unreviewed auto-assignment is a guess, and a guess must never enter
    the live gallery (only verified annotations are ground truth).
    Crops that cannot be embedded are dropped rather than reported.
    """
    embedded = []
    for annotation in annotations:
        if annotation.rejected or not annotation.verified or not annotation.crop_path:
            continue
        embedding = embed_crop_file(embedder, annotation.crop_path)
        if embedding is not None:
            embedded.append((annotation, embedding))
    return embedded


def enroll_embedded(vectors, identity, embedded, max_vectors: int = 20) -> int:
    """Add already-computed embeddings to an identity's gallery.

    Stops at the identifier's cap, and marks `enrolled` ONLY on the
    annotations whose vector actually landed: `enroll_verified` sweeps
    on `enrolled.is_(False)`, so flagging an annotation that was skipped
    for the cap would retire it from every future sweep — a label whose
    vector never arrives is exactly the failure this module exists to
    prevent. `identity.vector_count` is left to the caller, which reads
    it back from the store.
    """
    added = 0
    for annotation, embedding in embedded:
        if identity.vector_count + added >= max_vectors:
            break
        vectors.add(identity.identifier_key, embedding, identity.id)
        annotation.enrolled = True
        added += 1
    return added


def enroll_annotation(
    session: Session,
    annotation: Annotation,
    vectors,
    embedder,
    max_vectors: int = 20,
) -> bool:
    """Embed one verified annotation's crop under its identity.

    Returns True if a vector was added. Never raises on a bad crop —
    a face that cannot be re-embedded is logged and skipped, not fatal.
    """
    if annotation.enrolled or annotation.rejected or not annotation.verified:
        return False
    name = None
    if annotation.identity_id is not None:
        identity = session.get(Identity, annotation.identity_id)
        if identity is not None and identity.label:
            name = identity.label
    name = name or annotation.proposed_name
    if not name or not annotation.crop_path:
        return False

    identity = identity_for_label(session, name)
    annotation.identity_id = identity.id
    if identity.vector_count >= max_vectors:
        # The gallery is full for this person; the annotation is still
        # marked so sweeps don't reconsider it forever.
        annotation.enrolled = True
        return False

    embedding = embed_crop_file(embedder, annotation.crop_path)
    if embedding is None:
        log.debug("could not embed %s", annotation.crop_path)
        annotation.enrolled = True  # do not retry a hopeless crop
        return False

    vectors.add("face", embedding, identity.id)
    identity.vector_count += 1
    identity.appearance_count += 1
    identity.last_seen = _now()
    if annotation.crop_path and not identity.best_crop_path:
        identity.best_crop_path = annotation.crop_path
    annotation.enrolled = True
    return True


def enroll_verified(
    session: Session,
    vectors,
    embedder,
    max_vectors: int = 20,
    progress=None,
) -> EnrollStats:
    """Sweep all verified, unenrolled face annotations into the store."""
    stats = EnrollStats()
    rows = session.scalars(
        select(Annotation).filter(
            Annotation.class_name == "face",
            Annotation.verified.is_(True),
            Annotation.rejected.is_(False),
            Annotation.enrolled.is_(False),
        )
    ).all()
    seen_identities: set[int] = set()
    for i, annotation in enumerate(rows, 1):
        before = annotation.enrolled
        added = enroll_annotation(session, annotation, vectors, embedder, max_vectors)
        if added:
            stats.enrolled += 1
            seen_identities.add(annotation.identity_id)
        elif annotation.enrolled and not before:
            stats.skipped_cap += 1  # marked done without adding (cap/bad crop)
        if progress is not None:
            progress.advance(enrolled=1 if added else 0)
        if i % 200 == 0:
            session.commit()
            if progress is not None:
                progress.check_interrupt()
    session.commit()
    stats.identities = len(seen_identities)
    return stats
