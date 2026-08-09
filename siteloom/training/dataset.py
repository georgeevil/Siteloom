"""Training-data collection from verified annotations.

The invariant everything downstream depends on: **only verified,
non-rejected annotations are training data.** Auto-assignments proposed
by the Takeout importer are explicitly excluded until a human confirms
them in the review UI — otherwise the model would be trained on its own
guesses and the evaluation numbers would be meaningless.

There is a second sense of "confirmed" hiding inside the first, which is
why `human_only` exists (CLD-95). The importer's pass 1 auto-verifies
one-face-one-tag matches with nobody looking, and those rows are verified
in exactly the same column as the ones a person clicked through. The
default set is unchanged — every verified, non-rejected row, as before —
and `human_only=True` narrows it to `verified_by="human"`, so a fine-tune
can state what it trained on instead of assuming.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.store import VERIFIED_BY_HUMAN, Annotation, Identity, LibraryItem

log = logging.getLogger(__name__)


@dataclass
class FaceSample:
    person: str
    image_path: str
    bbox: list[float]  # normalized [x1, y1, x2, y2]
    crop_path: str | None
    annotation_id: int
    #: Who signed the annotation off — VERIFIED_BY_HUMAN, ..._IMPORT, or
    #: None on rows verified before the column existed and never
    #: attributed. Carried on the sample so a caller can report the
    #: composition of a set it already collected, without a second query.
    verified_by: str | None = None


def collect_face_samples(
    session: Session, min_per_person: int = 1, human_only: bool = False
) -> list[FaceSample]:
    """Verified face annotations with a resolved person name.

    The name comes from the linked Identity's label when there is one
    (the operator renamed it), else from the confirmed proposed_name.

    `human_only` restricts the set to annotations a person signed off on.
    It is an option, not the default: what counts as training data is
    unchanged by CLD-95, and machine-verified samples may well be good
    enough to train on — the point is being able to tell.
    """
    filters = [
        Annotation.class_name == "face",
        Annotation.verified.is_(True),
        Annotation.rejected.is_(False),
    ]
    if human_only:
        # Read, never inferred: pre-column rows were attributed once at
        # migration time (store/db.py), so a NULL here now means a row
        # this project did not write, and an unattributed row is not
        # evidence of human sign-off.
        filters.append(Annotation.verified_by == VERIFIED_BY_HUMAN)
    rows = session.scalars(
        select(Annotation)
        .join(LibraryItem, Annotation.item_id == LibraryItem.id)
        .filter(*filters)
    ).all()

    samples: list[FaceSample] = []
    for annotation in rows:
        person = None
        if annotation.identity_id is not None:
            identity = session.get(Identity, annotation.identity_id)
            if identity is not None and identity.label:
                person = identity.label
        person = person or annotation.proposed_name
        if not person:
            continue
        item = session.get(LibraryItem, annotation.item_id)
        if item is None or not Path(item.path).exists():
            continue
        samples.append(
            FaceSample(
                person=person,
                image_path=item.path,
                bbox=json.loads(annotation.bbox),
                crop_path=annotation.crop_path,
                annotation_id=annotation.id,
                verified_by=annotation.verified_by,
            )
        )

    if min_per_person > 1:
        counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            counts[sample.person] += 1
        dropped = {p for p, n in counts.items() if n < min_per_person}
        if dropped:
            log.info(
                "excluding %d people with fewer than %d samples: %s",
                len(dropped),
                min_per_person,
                ", ".join(sorted(dropped)[:5]) + ("…" if len(dropped) > 5 else ""),
            )
        samples = [s for s in samples if s.person not in dropped]
    return samples


def sample_provenance(samples: list[FaceSample]) -> dict[str, int]:
    """How many of these samples each verifier signed off on.

    Keys are the `verified_by` vocabulary plus "unattributed" for rows
    with none — a database this project did not write, or one whose
    migration never ran. Unattributed is reported as its own bucket
    rather than folded into either side: "we do not know" is not
    "a machine did it", and it is certainly not human sign-off.
    """
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.verified_by or "unattributed"] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def describe_provenance(samples: list[FaceSample]) -> str:
    """One line naming what a set is made of, for a run's notes."""
    counts = sample_provenance(samples)
    if not counts:
        return "no samples"
    return ", ".join(f"{n} {who}-verified" for who, n in counts.items())


def split_by_person(
    samples: list[FaceSample], val_fraction: float = 0.25
) -> tuple[list[FaceSample], list[FaceSample]]:
    """Split within each person, not across people.

    Face recognition is evaluated on *known* people seen in new photos, so
    every person must appear in both halves; a random global split would
    put some people entirely in validation and measure the wrong thing.

    Verification metrics also need **same-person pairs** in validation, so
    a person with enough samples contributes at least two. One-per-person
    would yield a validation set of all-different pairs, on which AUC is
    undefined — and an undefined score must never be mistaken for a good
    one (see evaluate_embeddings' `valid` flag).
    """
    by_person: dict[str, list[FaceSample]] = defaultdict(list)
    for sample in samples:
        by_person[sample.person].append(sample)

    train: list[FaceSample] = []
    val: list[FaceSample] = []
    for _person, person_samples in sorted(by_person.items()):
        ordered = sorted(person_samples, key=lambda s: s.annotation_id)
        n = len(ordered)
        if n < 2:
            n_val = 0  # never strand a person with zero training samples
        elif n < 4:
            n_val = 1
        else:
            n_val = max(2, int(n * val_fraction))
        n_val = min(n_val, n - 1)
        val.extend(ordered[:n_val])
        train.extend(ordered[n_val:])
    return train, val
