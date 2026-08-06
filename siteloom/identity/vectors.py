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

    def close(self) -> None:
        self._client.close()
