"""Custom sub-classes: operator-defined refinements of detection classes.

A detector gives you "car". You often want "delivery-van", "guest-vehicle",
"my-truck". Rather than retraining a detector for each, a custom class is
just a labeled set of example crops, classified by k-NN over the SAME
appearance embeddings the identity layer already computes.

That choice buys three things: defining a class costs one labeling pass
and zero training time; adding an example improves it immediately; and
the vector store and embedders are already in place, so there is no new
model to ship or keep in sync.

Examples live in the vector store under collection `class-examples`, with
the class name in the payload; the CustomClass row holds the threshold
and the parent detection class it refines.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.store import CustomClass

log = logging.getLogger(__name__)

COLLECTION = "class-examples"


@dataclass
class ClassVote:
    name: str
    score: float
    votes: int


class CustomClassifier:
    def __init__(self, vectors, k: int = 5):
        self.vectors = vectors
        self.k = k

    def add_example(self, vector: np.ndarray, class_name: str) -> None:
        """Register a labeled crop as an example of a custom class."""
        self.vectors.add_labeled(COLLECTION, vector, {"class_name": class_name})

    def classify(
        self, session: Session, vector: np.ndarray, parent_class: str = ""
    ) -> ClassVote | None:
        """k-NN vote over stored examples.

        The winner must clear its own configured threshold, so a
        deliberately strict class ("my-truck") and a loose one
        ("delivery-van") can coexist without one swamping the other.
        """
        hits = self.vectors.search_labeled(COLLECTION, vector, limit=self.k)
        if not hits:
            return None

        scores: dict[str, list[float]] = defaultdict(list)
        for hit in hits:
            name = hit.payload.get("class_name")
            if name:
                scores[name].append(hit.score)

        classes = {
            c.name: c
            for c in session.scalars(select(CustomClass)).all()
            if not parent_class or not c.parent_class or c.parent_class == parent_class
        }

        best: ClassVote | None = None
        for name, values in scores.items():
            custom = classes.get(name)
            if custom is None:
                continue  # class was deleted; its examples are ignored
            # Mean of that class's hits — one lucky neighbour shouldn't win.
            score = float(np.mean(values))
            if score < custom.threshold:
                continue
            if best is None or (len(values), score) > (best.votes, best.score):
                best = ClassVote(name=name, score=score, votes=len(values))
        return best

    def rebuild(self, session: Session, embed_crop) -> int:
        """Re-derive all examples from verified annotations.

        Called after labeling sessions and after the embedder changes
        (e.g. a fine-tuned face projection lands) — stale vectors from an
        old embedding space would otherwise silently degrade voting.
        """
        from siteloom.store import Annotation

        self.vectors.drop(COLLECTION)
        count = 0
        annotations = session.scalars(
            select(Annotation).filter(
                Annotation.custom_class.is_not(None),
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
            )
        ).all()
        counts: dict[str, int] = defaultdict(int)
        for annotation in annotations:
            if not annotation.crop_path:
                continue
            vector = embed_crop(annotation.crop_path)
            if vector is None:
                continue
            self.add_example(vector, annotation.custom_class)
            counts[annotation.custom_class] += 1
            count += 1
        for custom in session.scalars(select(CustomClass)).all():
            custom.example_count = counts.get(custom.name, 0)
        session.commit()
        log.info("rebuilt %d custom-class examples", count)
        return count
