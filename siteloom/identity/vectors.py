"""Local vector database (PRD §6.7, vector half).

Qdrant in embedded/local mode: `QdrantClient(path=...)` runs the engine
in-process and persists to a directory — no server, no docker. The same
client class speaks to a remote Qdrant later, so moving the vector store
to a central platform (V1 multi-site) is a config change.

One collection per identifier key ("face", "person", "vehicle", plus any
dynamically added class). Collections are created on demand — this is
what lets new classes appear at runtime without a schema migration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class Hit:
    identity_id: int
    score: float  # cosine similarity, higher = closer


@dataclass
class LabeledHit:
    """A hit carrying an arbitrary payload — used by the custom-class
    classifier, which votes on payload fields rather than identity ids."""

    payload: dict
    score: float


class VectorStore:
    def __init__(self, path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))

    def ensure_collection(self, name: str, dim: int) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, collection: str, vector: np.ndarray, identity_id: int) -> None:
        self.ensure_collection(collection, dim=int(vector.shape[0]))
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector.astype(np.float32).tolist(),
                    payload={"identity_id": identity_id},
                )
            ],
        )

    def search(self, collection: str, vector: np.ndarray, limit: int = 5) -> list[Hit]:
        if not self._client.collection_exists(collection):
            return []
        res = self._client.query_points(
            collection_name=collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
        )
        return [
            Hit(identity_id=int(p.payload["identity_id"]), score=float(p.score))
            for p in res.points
        ]

    def best_match(self, collection: str, vector: np.ndarray) -> Hit | None:
        """Best identity by max similarity over its stored embeddings."""
        hits = self.search(collection, vector, limit=5)
        return hits[0] if hits else None

    # -- payload-keyed API (custom classes) --------------------------------

    def add_labeled(
        self, collection: str, vector: np.ndarray, payload: dict
    ) -> None:
        self.ensure_collection(collection, dim=int(vector.shape[0]))
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector.astype(np.float32).tolist(),
                    payload=payload,
                )
            ],
        )

    def search_labeled(
        self, collection: str, vector: np.ndarray, limit: int = 5
    ) -> list[LabeledHit]:
        if not self._client.collection_exists(collection):
            return []
        res = self._client.query_points(
            collection_name=collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
        )
        return [
            LabeledHit(payload=dict(p.payload or {}), score=float(p.score))
            for p in res.points
        ]

    def drop(self, collection: str) -> None:
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)

    def delete_identity(self, collection: str, identity_id: int) -> None:
        """Remove every vector belonging to an identity — used when
        identities are merged or split."""
        if not self._client.collection_exists(collection):
            return
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="identity_id", match=MatchValue(value=identity_id)
                    )
                ]
            ),
        )

    def reassign_identity(self, collection: str, old_id: int, new_id: int) -> int:
        """Move all of one identity's vectors to another (merge)."""
        if not self._client.collection_exists(collection):
            return 0
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        selector = Filter(
            must=[FieldCondition(key="identity_id", match=MatchValue(value=old_id))]
        )
        points, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=selector,
            limit=10_000,
            with_payload=True,
        )
        if not points:
            return 0
        self._client.set_payload(
            collection_name=collection,
            payload={"identity_id": new_id},
            points=[p.id for p in points],
        )
        return len(points)

    def close(self) -> None:
        self._client.close()
