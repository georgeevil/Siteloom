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

import threading
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def _locked(method):
    """Serialize access to the embedded engine.

    Qdrant local mode is not thread-safe; a FastAPI threadpool (the
    recognition API) or any future worker threads would corrupt it.
    Combined with force_disable_check_same_thread below, a single lock
    makes cross-thread use safe — contention is irrelevant at PoC scale.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


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
        self._lock = threading.RLock()
        self._client = QdrantClient(
            path=str(path), force_disable_check_same_thread=True
        )

    @_locked
    def ensure_collection(self, name: str, dim: int) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    @_locked
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

    @_locked
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

    @_locked
    def best_match(self, collection: str, vector: np.ndarray) -> Hit | None:
        """Best identity by max similarity over its stored embeddings."""
        hits = self.search(collection, vector, limit=5)
        return hits[0] if hits else None

    # -- payload-keyed API (custom classes) --------------------------------

    @_locked
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

    @_locked
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

    @_locked
    def pop_matching(
        self,
        collection: str,
        vector: np.ndarray,
        threshold: float,
        limit: int = 64,
    ) -> list[tuple[np.ndarray, dict]]:
        """Remove and return every point similar to `vector` at or above
        `threshold` — the pending-pool promotion step (CLD-41): the
        cluster's vectors move out of quarantine and into an identity's
        collection, so they must leave here atomically with the read."""
        if not self._client.collection_exists(collection):
            return []
        res = self._client.query_points(
            collection_name=collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        matched = [p for p in res.points if float(p.score) >= threshold]
        if not matched:
            return []
        self._client.delete(
            collection_name=collection,
            points_selector=[p.id for p in matched],
        )
        return [
            (np.asarray(p.vector, dtype=np.float32), dict(p.payload or {}))
            for p in matched
        ]

    @_locked
    def prune_older_than(self, collection: str, cutoff_ts: float) -> None:
        """Delete points whose payload "ts" is before `cutoff_ts` (epoch
        seconds). Keeps the pending pool from accumulating one-off crops."""
        if not self._client.collection_exists(collection):
            return
        from qdrant_client.models import FieldCondition, Filter, Range

        self._client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="ts", range=Range(lt=cutoff_ts))]
            ),
        )

    @_locked
    def drop(self, collection: str) -> None:
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)

    @_locked
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

    @_locked
    def count_identity(self, collection: str, identity_id: int) -> int:
        """How many vectors an identity actually has in the store.

        Identity.vector_count is maintained by increments across several
        writers and can drift; this is the authority when a count has to
        be correct (identity split rewrites both sides from it)."""
        if not self._client.collection_exists(collection):
            return 0
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return int(
            self._client.count(
                collection_name=collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="identity_id", match=MatchValue(value=identity_id)
                        )
                    ]
                ),
            ).count
        )

    @_locked
    def delete_duplicates_of(
        self,
        collection: str,
        identity_id: int,
        vectors: list[np.ndarray],
        min_score: float = 0.999,
    ) -> int:
        """Delete an identity's vectors that duplicate one of `vectors`.

        Provenance-free surgery for the identity split. Vector payloads
        record only `identity_id` — nothing says which stored vector came
        from which annotation — so an identity's gallery cannot be
        separated by origin. What *can* be established is numerical
        identity: re-embedding a stored crop with the same embedder
        reproduces the vector that crop contributed (deterministic
        embedder, same image — "one crop, two jobs"), so a stored vector
        at cosine ≈ 1.0 to a re-embedded crop IS that crop's vector.

        This deletes only those provable duplicates and leaves every
        other vector alone — crucially the ones live camera matching
        added, which have no annotation to rebuild them from and would
        be destroyed by a wholesale delete-and-rebuild.

        Vectors from the embedders are L2-normalized, so the dot product
        is the cosine similarity. Returns the number deleted.
        """
        if not vectors or not self._client.collection_exists(collection):
            return 0
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="identity_id", match=MatchValue(value=identity_id)
                    )
                ]
            ),
            limit=10_000,
            with_vectors=True,
        )
        reference = np.asarray(vectors, dtype=np.float32)
        doomed = []
        for point in points:
            if point.vector is None:
                continue
            candidate = np.asarray(point.vector, dtype=np.float32)
            if candidate.shape[0] != reference.shape[1]:
                continue  # a different embedding space; not ours to judge
            if float(np.max(reference @ candidate)) >= min_score:
                doomed.append(point.id)
        if doomed:
            self._client.delete(collection_name=collection, points_selector=doomed)
        return len(doomed)

    @_locked
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

    @_locked
    def close(self) -> None:
        self._client.close()


#: One shared client per storage path per process. Embedded Qdrant takes
#: an exclusive flock on `<path>/.lock`, and a second client on the same
#: path fails with a RuntimeError even when opened by the SAME process
#: (the lock is per open-file-description). Every in-process consumer
#: (recognition API, face enrollment, identity merge) must therefore
#: reuse one instance — see CLAUDE.md's "one client per path per
#: machine" rule. The `_locked` wrapper already serializes calls across
#: FastAPI's threadpool, which is what makes sharing safe.
_shared_stores: dict[str, VectorStore] = {}
_shared_stores_lock = threading.Lock()


def get_shared_store(path: str | Path) -> VectorStore:
    """Return the process-wide VectorStore for `path`, creating it on
    first use. Never close the result — other components hold the same
    reference, and the OS releases the lock at process exit."""
    key = str(Path(path).resolve())
    with _shared_stores_lock:
        store = _shared_stores.get(key)
        if store is None:
            store = VectorStore(path)
            _shared_stores[key] = store
        return store
