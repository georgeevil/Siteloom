"""Training-data collection from verified annotations.

The invariant everything downstream depends on: **only verified,
non-rejected annotations are training data.** Auto-assignments proposed
by the Takeout importer are explicitly excluded until a human confirms
them in the review UI — otherwise the model would be trained on its own
guesses and the evaluation numbers would be meaningless.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.store import Annotation, Identity, LibraryItem

log = logging.getLogger(__name__)


@dataclass
class FaceSample:
    person: str
    image_path: str
    bbox: list[float]  # normalized [x1, y1, x2, y2]
    crop_path: str | None
    annotation_id: int


def collect_face_samples(
    session: Session, min_per_person: int = 1
) -> list[FaceSample]:
    """Verified face annotations with a resolved person name.

    The name comes from the linked Identity's label when there is one
    (the operator renamed it), else from the confirmed proposed_name.
    """
    rows = session.scalars(
        select(Annotation)
        .join(LibraryItem, Annotation.item_id == LibraryItem.id)
        .filter(
            Annotation.class_name == "face",
            Annotation.verified.is_(True),
            Annotation.rejected.is_(False),
        )
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
