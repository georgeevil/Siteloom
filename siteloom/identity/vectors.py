"""Local vector database (PRD §6.7, vector half).

Qdrant in embedded/local mode: `QdrantClient(path=...)` runs the engine
in-process and persists to a directory — no server, no docker. The same
client class speaks to a remote Qdrant later, so moving the vector store
to a central platform (V1 multi-site) is a config change.

One collection per identifier key ("face", "person", "vehicle", plus any
dynamically added class). Collections are created on demand — this is
what lets new classes appear at runtime without a schema migration.

Every point carries provenance (CLD-84): which writer added it and what
it was made from. An identity's gallery is several disjoint populations
— live camera matches, enrolled library annotations, operator
corrections, API enrollments — and without a marker on the payload they
are indistinguishable, which forces every surgical edit (split, unlink,
reassign) to reason numerically about vectors instead of by origin. With
it, "delete the vectors this annotation contributed" and "move the
vectors this event contributed" are exact queries. Points written before
provenance existed carry no marker; they are reachable only through
`delete_duplicates_of`, and acquire provenance on the next re-enroll
(already a documented re-enroll event, like changing the embedder).
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


#: Payload `source` markers — who wrote the vector (CLD-84).
#: "live" is the only population with no on-disk record to rebuild it
#: from, which is why it must never be swept up by a wholesale rebuild.
SOURCE_LIVE = "live"  # identity/resolver.py, during camera matching
SOURCE_ENROLLED = "enrolled"  # identity/enroll.py, from a verified annotation
SOURCE_MANUAL = "manual"  # an operator correction in the console (CLD-36)
SOURCE_API = "api"  # the CompreFace-compatible enrollment endpoint


#: Raw-point window for identity candidate search. Wide enough that a
#: healthy gallery (max_vectors_per_identity defaults to 20) rarely
#: monopolises it, cheap enough to run per identified frame: flat search
#: cost is ~1 ms at 1k points, ~3 ms at 5k, and barely moves with limit.
CANDIDATE_POINTS = 32


@dataclass
class Hit:
    identity_id: int
    score: float  # cosine similarity, higher = closer


def _ranked_hits(best: dict[int, float]) -> list[Hit]:
    return [
        Hit(identity_id=identity_id, score=score)
        for identity_id, score in sorted(
            best.items(), key=lambda kv: kv[1], reverse=True
        )
    ]


@dataclass
class LabeledHit:
    """A hit carrying an arbitrary payload — used by the custom-class
    classifier, which votes on payload fields rather than identity ids."""

    payload: dict
    score: float


class VectorStore:
    def __init__(self, path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._path_key = str(Path(path).resolve())
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
    def add(
        self,
        collection: str,
        vector: np.ndarray,
        identity_id: int,
        *,
        source: str = SOURCE_LIVE,
        annotation_id: int | None = None,
        crop_path: str | None = None,
    ) -> None:
        """Store one embedding under an identity, with its provenance.

        `source` says which writer added it; `annotation_id` and
        `crop_path` say what it was made from, where that is known. Both
        are the handles later surgery uses to find this exact vector
        again — an annotation moved to another identity takes its vector
        with it, an event's claim that gets reassigned takes the vectors
        its crops contributed. Recording them costs a dict key at write
        time and is the difference between an exact edit and a numeric
        guess (CLD-84).
        """
        self.ensure_collection(collection, dim=int(vector.shape[0]))
        payload: dict = {"identity_id": identity_id, "source": source}
        if annotation_id is not None:
            payload["annotation_id"] = int(annotation_id)
        if crop_path:
            payload["crop_path"] = str(crop_path)
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
    def search_identities(
        self,
        collection: str,
        vector: np.ndarray,
        *,
        limit: int = CANDIDATE_POINTS,
        min_identities: int = 2,
    ) -> list[Hit]:
        """Best score per *identity*, ranked — with the runner-up guaranteed.

        The contest the resolver runs is between individuals, not between
        raw vectors, and the guarantee it needs is that if two identities
        have any vector in this collection, it sees two. A flat top-k does
        not provide that: one identity holding k near-duplicates fills the
        window, `ranked` collapses to length 1, and `min_margin` — the
        check that exists precisely for a crowded neighbourhood — is
        skipped where it matters most (CLD-139).

        Two passes, because the exact answer is 20x the price of the
        common one (measured: 12 ms vs 1 ms at 1k points, 62 ms vs 3 ms at
        5k):

        1. Flat search at `limit`, grouped in Python. If **fewer than
           `limit` points came back, this window is the whole
           collection** — there is no hidden identity, and the grouping is
           exhaustive by construction. That is the ordinary case and it is
           one search.
        2. Only when the window came back *saturated* AND collapsed to
           fewer than `min_identities` — the pathological case, one
           gallery monopolising the window — re-ask with Qdrant's grouped
           search, which groups server-side over the whole collection and
           is unaffected by gallery size (verified against a 2000-vector
           gallery).

        Returns one Hit per identity, best score first. No fallback path
        if grouped search is unavailable: the pinned client supports it in
        both local and server mode, and a silent degradation here would
        restore exactly the bug being fixed.
        """
        if not self._client.collection_exists(collection):
            return []
        res = self._client.query_points(
            collection_name=collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
        )
        best: dict[int, float] = {}
        for point in res.points:  # sorted by score desc
            best.setdefault(int(point.payload["identity_id"]), float(point.score))
        if len(res.points) < limit or len(best) >= min_identities:
            return _ranked_hits(best)

        # Verified working against embedded Qdrant, which needs no index
        # to group on a payload field. A remote server (V1 multi-site)
        # may want a payload index on identity_id for this to stay cheap
        # — nothing to do today, and the same call either way.
        groups = self._client.query_points_groups(
            collection_name=collection,
            query=vector.astype(np.float32).tolist(),
            group_by="identity_id",
            limit=min_identities,
            group_size=1,
        )
        return _ranked_hits(
            {
                int(point.payload["identity_id"]): float(point.score)
                for group in groups.groups
                for point in group.hits[:1]
            }
        )

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

    def _scroll_identity(
        self, collection: str, identity_id: int, *, with_vectors: bool = False
    ) -> list:
        """Every point an identity owns. Callers hold the lock.

        Filtering by identity in the engine and by payload in Python is
        deliberate: the identity filter is the safety property (a
        bystander's gallery must never be touched — see
        `test_split_does_not_touch_another_identitys_matching_vectors`),
        while the payload predicates vary per caller and are cheap over
        one identity's handful of vectors.
        """
        if not self._client.collection_exists(collection):
            return []
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
            with_payload=True,
            with_vectors=with_vectors,
        )
        return points

    @_locked
    def delete_by_annotations(
        self, collection: str, identity_id: int, annotation_ids
    ) -> int:
        """Delete the vectors an identity got from these annotations.

        The exact form of what `delete_duplicates_of` approximates: a
        point written by `identity/enroll.py` records the annotation it
        came from, so moving that annotation elsewhere can take its
        vector with it by origin rather than by numeric coincidence.
        Only points carrying provenance are eligible — pre-CLD-84 points
        have no `annotation_id` and stay for the numeric pass. Returns
        the number deleted.
        """
        wanted = {int(a) for a in annotation_ids}
        if not wanted:
            return 0
        doomed = [
            p.id
            for p in self._scroll_identity(collection, identity_id)
            if (p.payload or {}).get("annotation_id") in wanted
        ]
        if doomed:
            self._client.delete(collection_name=collection, points_selector=doomed)
        return len(doomed)

    @_locked
    def delete_by_crops(
        self, collection: str, identity_id: int, crop_paths
    ) -> int:
        """Delete the vectors an identity got from these crop files.

        The handle for live-matched vectors, which have no annotation:
        the resolver records the crop it embedded ("one crop, two jobs"),
        so an operator repudiating an event's claim can strip exactly the
        vectors that event taught the wrong identity — the pollution that
        otherwise keeps re-attracting the same wrong match.
        """
        wanted = {str(p) for p in crop_paths if p}
        if not wanted:
            return 0
        doomed = [
            p.id
            for p in self._scroll_identity(collection, identity_id)
            if (p.payload or {}).get("crop_path") in wanted
        ]
        if doomed:
            self._client.delete(collection_name=collection, points_selector=doomed)
        return len(doomed)

    @_locked
    def move_by_crops(
        self, collection: str, old_id: int, new_id: int, crop_paths
    ) -> int:
        """Re-point the vectors made from these crops onto another identity.

        A reassignment (CLD-36) is not a deletion: the crops really are
        the new identity, and the human saying so is stronger evidence
        than any cosine score — so the vectors move rather than being
        thrown away, correcting the wrong gallery and teaching the right
        one in one step.
        """
        wanted = {str(p) for p in crop_paths if p}
        if not wanted:
            return 0
        moving = [
            p.id
            for p in self._scroll_identity(collection, old_id)
            if (p.payload or {}).get("crop_path") in wanted
        ]
        if moving:
            self._client.set_payload(
                collection_name=collection,
                payload={"identity_id": new_id},
                points=moving,
            )
        return len(moving)

    @_locked
    def delete_duplicates_of(
        self,
        collection: str,
        identity_id: int,
        vectors: list[np.ndarray],
        min_score: float = 0.999,
    ) -> int:
        """Delete an identity's vectors that duplicate one of `vectors`.

        The legacy pass, for points written before payloads carried
        provenance (CLD-84). Those record only `identity_id` — nothing
        says which stored vector came from which annotation — so that
        part of a gallery cannot be separated by origin. What *can* be
        established is numerical identity: re-embedding a stored crop
        with the same embedder reproduces the vector that crop
        contributed (deterministic embedder, same image — "one crop, two
        jobs"), so a stored vector at cosine ≈ 1.0 to a re-embedded crop
        IS that crop's vector.

        `delete_by_annotations` is the exact form and runs first; this
        catches whatever it could not identify. It deletes only provable
        duplicates and leaves every other vector alone — crucially the
        ones live camera matching added, which have no annotation to
        rebuild them from and would be destroyed by a wholesale
        delete-and-rebuild.

        Vectors from the embedders are L2-normalized, so the dot product
        is the cosine similarity. Returns the number deleted.
        """
        if not vectors:
            return 0
        points = self._scroll_identity(collection, identity_id, with_vectors=True)
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
        points = self._scroll_identity(collection, old_id)
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
        """Close the embedded client and release its lock.

        A closed client is unusable, so if this instance is the process's
        shared store it is also evicted from the cache — the next
        get_shared_store() opens a fresh client instead of handing out a
        dead one. Live services never close the shared store (the lock is
        released at process exit); closing is for tests that simulate a
        process restart on one path, and for deliberate shutdowns.
        """
        with _shared_stores_lock:
            if _shared_stores.get(self._path_key) is self:
                del _shared_stores[self._path_key]
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
