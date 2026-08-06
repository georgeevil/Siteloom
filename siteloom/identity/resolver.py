"""Identity resolution: embedding -> existing or new Identity.

Owns all identity state (vector store + Identity rows). Runs in the
application layer next to the databases — on a distributed deployment
this stays central while IdentityModule runs at the edge.

Matching policy per identifier:
1. Plate first (vehicles, PRD §6.4): an OCR'd plate that matches an
   existing identity's plate wins outright — plates are stronger
   evidence than visual similarity.
2. Otherwise best cosine hit >= the identifier's threshold -> match.
3. Otherwise a new Identity row is created (label=None, the
   label-and-learn "unknown" bucket, PRD §6.3).

On every resolution the embedding is added to the identity's collection
(capped) so matches keep improving as an identity accumulates views.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from siteloom.config import IdentityConfig
from siteloom.identity.vectors import VectorStore
from siteloom.store.models import Identity


@dataclass
class Resolution:
    identity: Identity
    similarity: float
    is_new: bool


class IdentityResolver:
    def __init__(self, cfg: IdentityConfig, vectors: VectorStore):
        self.cfg = cfg
        self.vectors = vectors

    def resolve(
        self,
        session: Session,
        *,
        identifier_key: str,
        class_name: str,
        vector: list[float] | None,
        plate: str | None,
        timestamp: datetime,
        crop_path: str | None = None,
        threshold: float | None = None,
        max_vectors: int = 20,
    ) -> Resolution:
        arr = np.asarray(vector, dtype=np.float32) if vector is not None else None

        identity, similarity = self._match_plate(session, identifier_key, plate)
        if identity is None and arr is not None:
            identity, similarity = self._match_vector(
                session, identifier_key, arr, threshold
            )

        is_new = identity is None
        if is_new:
            identity = Identity(
                identifier_key=identifier_key,
                class_name=class_name,
                plate=plate,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            session.add(identity)
            session.flush()  # assign id for the vector payload

        # Update stats + evidence.
        identity.last_seen = max(identity.last_seen, timestamp)
        identity.appearance_count += 1
        if plate and not identity.plate:
            identity.plate = plate  # visual match just learned its plate
        if crop_path and not identity.best_crop_path:
            identity.best_crop_path = crop_path
        if arr is not None and identity.vector_count < max_vectors:
            self.vectors.add(identifier_key, arr, identity.id)
            identity.vector_count += 1

        return Resolution(identity=identity, similarity=similarity, is_new=is_new)

    def _match_plate(
        self, session: Session, identifier_key: str, plate: str | None
    ) -> tuple[Identity | None, float]:
        if not plate:
            return None, 0.0
        identity = (
            session.query(Identity)
            .filter_by(identifier_key=identifier_key, plate=plate)
            .first()
        )
        return identity, 1.0 if identity else 0.0

    def _match_vector(
        self,
        session: Session,
        identifier_key: str,
        arr: np.ndarray,
        threshold: float | None,
    ) -> tuple[Identity | None, float]:
        if threshold is None:
            ident_cfg = self.cfg.identifiers.get(identifier_key)
            threshold = ident_cfg.threshold if ident_cfg else self.cfg.auto_add_threshold
        hit = self.vectors.best_match(identifier_key, arr)
        if hit is None or hit.score < threshold:
            return None, hit.score if hit else 0.0
        return session.get(Identity, hit.identity_id), hit.score
