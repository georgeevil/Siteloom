"""Identity resolution: embedding -> existing or new Identity.

Owns all identity state (vector store + Identity rows). Runs in the
application layer next to the databases — on a distributed deployment
this stays central while IdentityModule runs at the edge.

Matching policy per identifier (CLD-41 hardened):
1. Plate first (vehicles, PRD §6.4): an OCR'd plate that matches an
   existing identity's plate wins outright — plates are stronger
   evidence than visual similarity.
2. Otherwise the best cosine candidate must clear the identifier's
   threshold AND beat the runner-up *identity* by `min_margin`. Two
   identities inside the margin band are an ambiguous read: the
   camera-recency prior may break the tie (an identity seen on the same
   camera within `recency_window_s` wins), otherwise the frame is left
   unresolved rather than guessed. Ambiguity between known identities
   never creates a third one.
3. Otherwise the embedding is an unknown. With `min_sightings` > 1 it is
   parked in the identifier's pending pool (`<key>-pending` collection)
   and an Identity row is only minted once that many mutually-similar
   sightings accumulate — a single weak crop must not become a
   vector-space magnet. A plate, or detection quality at or above
   `immediate_quality`, mints immediately.

On every resolution that yields an identity, the embedding is added to
its collection (capped) so matches keep improving as the identity
accumulates views.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from siteloom.config import IdentifierConfig, IdentityConfig
from siteloom.identity.vectors import VectorStore
from siteloom.store.models import Identity


def _pending_collection(identifier_key: str) -> str:
    return f"{identifier_key}-pending"


@dataclass
class Resolution:
    # None when the frame did not resolve: the embedding was parked in
    # the pending pool (`pending`) or sat between two known identities
    # (`ambiguous`). Callers create no EventIdentity link in either case.
    identity: Identity | None
    similarity: float
    is_new: bool
    # An existing identity, previously plate-less, just gained a plate
    # from OCR on this pass (PRD §6.4's "visual match learns its plate").
    learned_plate: bool = False
    # How the match was made: "plate" or "visual"; None for a new
    # identity. Persisted onto EventIdentity so plate-vs-visual accuracy
    # is a query, not a webhook-log archaeology exercise (CLD-17).
    matched_by: str | None = None
    # Unknown embedding quarantined until min_sightings accumulate.
    pending: bool = False
    # Top-two identities inside the margin band and no recency winner.
    ambiguous: bool = False


class IdentityResolver:
    def __init__(self, cfg: IdentityConfig, vectors: VectorStore):
        self.cfg = cfg
        self.vectors = vectors
        # Camera-recency prior: identifier key -> identity id -> camera
        # id -> last stream timestamp. In-memory on purpose — it is a
        # soft tie-break, and losing it on restart degrades to the
        # conservative no-match, never to a wrong match.
        self._recent: dict[str, dict[int, dict[str, datetime]]] = {}

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
        camera_id: str | None = None,
        quality: float | None = None,
    ) -> Resolution:
        arr = np.asarray(vector, dtype=np.float32) if vector is not None else None
        ident_cfg = self.cfg.identifiers.get(identifier_key)

        identity, similarity = self._match_plate(session, identifier_key, plate)
        matched_by = "plate" if identity is not None else None
        ambiguous = False
        if identity is None and arr is not None:
            identity, similarity, ambiguous = self._match_vector(
                session, identifier_key, ident_cfg, arr, threshold,
                camera_id, timestamp,
            )
            if identity is not None:
                matched_by = "visual"

        if identity is None and ambiguous:
            # Between two known identities: report, learn nothing. Adding
            # the vector to either gallery — or to the pending pool,
            # where it could seed a duplicate — would compound the error.
            return Resolution(
                identity=None, similarity=similarity, is_new=False,
                ambiguous=True,
            )

        is_new = identity is None
        promoted_first_seen = timestamp
        promoted: list[tuple[np.ndarray, dict]] = []
        if is_new:
            min_sightings = ident_cfg.min_sightings if ident_cfg else 1
            gate = (
                min_sightings > 1
                and arr is not None
                and plate is None  # a plate is exact evidence: mint now
                and (quality is None
                     or quality < (ident_cfg.immediate_quality if ident_cfg else 0.85))
            )
            if gate:
                eff_threshold = self._threshold(identifier_key, ident_cfg, threshold)
                promoted = self._consistent_sightings(
                    identifier_key, arr, eff_threshold, min_sightings,
                    camera_id, timestamp, crop_path,
                )
                if promoted is None:
                    return Resolution(
                        identity=None, similarity=similarity, is_new=False,
                        pending=True,
                    )
                for _, payload in promoted:
                    ts = payload.get("ts")
                    if ts is not None:
                        promoted_first_seen = min(
                            promoted_first_seen,
                            datetime.fromtimestamp(float(ts)),
                        )
            identity = Identity(
                identifier_key=identifier_key,
                class_name=class_name,
                plate=plate,
                first_seen=promoted_first_seen,
                last_seen=timestamp,
            )
            session.add(identity)
            session.flush()  # assign id for the vector payload
            # The quarantined sightings become the new identity's gallery
            # and count as appearances — they were this individual all
            # along, the system just refused to guess from one of them.
            # Each carries the crop it was made from through quarantine,
            # so a promoted vector is as traceable as a directly matched
            # one (CLD-84).
            for vec, payload in promoted[: max(0, max_vectors - 1)]:
                self.vectors.add(
                    identifier_key,
                    vec,
                    identity.id,
                    crop_path=payload.get("crop_path"),
                )
                identity.vector_count += 1
            identity.appearance_count += len(promoted)

        # Update stats + evidence.
        identity.last_seen = max(identity.last_seen, timestamp)
        identity.appearance_count += 1
        learned_plate = bool(plate and not identity.plate and not is_new)
        if plate and not identity.plate:
            identity.plate = plate  # visual match just learned its plate
        if crop_path and not identity.best_crop_path:
            identity.best_crop_path = crop_path
        if arr is not None and identity.vector_count < max_vectors:
            # `crop_path` is this vector's provenance (CLD-84): the live
            # population has no Annotation row, so the crop file is the
            # only thing that can name it later — which is what lets an
            # operator's correction move or strip exactly the vectors one
            # event taught an identity.
            self.vectors.add(identifier_key, arr, identity.id, crop_path=crop_path)
            identity.vector_count += 1
        self._note_seen(identifier_key, identity.id, camera_id, timestamp)

        return Resolution(
            identity=identity,
            similarity=similarity,
            is_new=is_new,
            learned_plate=learned_plate,
            matched_by=matched_by,
        )

    def _threshold(
        self,
        identifier_key: str,
        ident_cfg: IdentifierConfig | None,
        threshold: float | None,
    ) -> float:
        if threshold is not None:
            return threshold
        return ident_cfg.threshold if ident_cfg else self.cfg.auto_add_threshold

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
        ident_cfg: IdentifierConfig | None,
        arr: np.ndarray,
        threshold: float | None,
        camera_id: str | None,
        timestamp: datetime,
    ) -> tuple[Identity | None, float, bool]:
        """Best identity by score, guarded by the margin + recency rules.

        Returns (identity, best score, ambiguous). Aggregation per
        *identity* (max over that identity's hits) happens in the store,
        which is also where the guarantee this method depends on lives:
        if two identities have vectors here, `ranked` has two entries.
        That is structural rather than a matter of asking for a bigger
        window — no flat top-k is large enough when one gallery can hold
        every slot, which is how the margin check came to be skipped in
        exactly the crowded neighbourhood it exists for (CLD-139).
        """
        eff_threshold = self._threshold(identifier_key, ident_cfg, threshold)
        margin = ident_cfg.min_margin if ident_cfg else 0.0
        ranked = [
            (hit.identity_id, hit.score)
            for hit in self.vectors.search_identities(identifier_key, arr)
        ]
        if not ranked or ranked[0][1] < eff_threshold:
            return None, ranked[0][1] if ranked else 0.0, False
        top_id, top_score = ranked[0]
        if len(ranked) > 1 and margin > 0.0:
            runner_id, runner_score = ranked[1]
            if top_score - runner_score < margin:
                pick = self._recency_pick(
                    identifier_key,
                    [(top_id, top_score), (runner_id, runner_score)],
                    eff_threshold, camera_id, timestamp,
                )
                if pick is None:
                    return None, top_score, True
                top_id, top_score = pick
        return session.get(Identity, top_id), top_score, False

    def _recency_pick(
        self,
        identifier_key: str,
        candidates: list[tuple[int, float]],
        threshold: float,
        camera_id: str | None,
        timestamp: datetime,
    ) -> tuple[int, float] | None:
        """Tie-break a margin failure: if exactly one of the contenders
        was seen on this camera within the recency window (and clears the
        threshold on its own), it wins. Bounded to the ambiguity band —
        this can never override a clear better match, because it is only
        consulted when there isn't one."""
        if camera_id is None:
            return None
        window = self.cfg.recency_window_s
        seen = self._recent.get(identifier_key, {})
        recent = [
            (identity_id, score)
            for identity_id, score in candidates
            if score >= threshold
            and (last := seen.get(identity_id, {}).get(camera_id)) is not None
            and abs((timestamp - last).total_seconds()) <= window
        ]
        return recent[0] if len(recent) == 1 else None

    def _note_seen(
        self,
        identifier_key: str,
        identity_id: int,
        camera_id: str | None,
        timestamp: datetime,
    ) -> None:
        if camera_id is None:
            return
        cams = self._recent.setdefault(identifier_key, {}).setdefault(
            identity_id, {}
        )
        prev = cams.get(camera_id)
        if prev is None or timestamp > prev:
            cams[camera_id] = timestamp

    def _consistent_sightings(
        self,
        identifier_key: str,
        arr: np.ndarray,
        threshold: float,
        min_sightings: int,
        camera_id: str | None,
        timestamp: datetime,
        crop_path: str | None = None,
    ) -> list[tuple[np.ndarray, dict]] | None:
        """Count this unknown against the identifier's pending pool.

        Returns the promoted cluster (vectors + payloads, removed from
        the pool) once `min_sightings` accumulate including the current
        one, or None after parking the embedding for later. The pool is
        pruned to `pending_ttl_s` on stream time first, so a one-off
        blurry crop expires instead of seeding a promotion forever.
        """
        pool = _pending_collection(identifier_key)
        self.vectors.prune_older_than(
            pool, timestamp.timestamp() - self.cfg.pending_ttl_s
        )
        prior = self.vectors.search_labeled(pool, arr, limit=max(min_sightings * 4, 16))
        similar = sum(1 for hit in prior if hit.score >= threshold)
        if similar + 1 < min_sightings:
            self.vectors.add_labeled(
                pool,
                arr,
                {
                    "ts": timestamp.timestamp(),
                    "camera": camera_id,
                    "crop_path": crop_path,
                },
            )
            return None
        return self.vectors.pop_matching(pool, arr, threshold)
