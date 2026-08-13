"""CompreFace-compatible face recognition REST API.

Tools in the Frigate ecosystem — Double Take above all — already speak
CompreFace's Recognition Service API. Implementing that surface means
any of them can point at Siteloom as their recognizer with only a URL
change, and get matches out of the SAME face collection the cameras and
the photo backfill share (a "subject" here is an Identity row's label).

Implemented endpoints (the subset Double Take and enrollment scripts
actually use), shape-compatible with CompreFace v1:

    POST /api/v1/recognition/recognize            multipart file -> matches
    GET  /api/v1/recognition/subjects             list subjects
    POST /api/v1/recognition/subjects             {"subject": name}
    POST /api/v1/recognition/faces?subject=name   multipart file -> enroll
    GET  /api/v1/recognition/faces                list enrolled examples

Auth follows CompreFace's convention: the x-api-key header, checked
against integrations.recognition_api.api_key when one is configured.
Serving without a key is an explicit opt-in (allow_open) — this surface
hands out biometric matching and gallery writes, so enabled-but-keyless
fails fast at startup instead of coming up wide open (CLD-47). The 413
upload cap and per-IP 429 are additions on top of the CompreFace v1
shape, not redefinitions of it.

While another process holds the embedded vector store — `siteloom run`
does, for hours, which is the NORMAL condition on a live site — reads
degrade and writes refuse (CLD-110). /recognize answers 200 in its
ordinary shape with zero subjects, plus an additive "degraded" marker
and a Retry-After header, so a polling Double Take reports "unknown"
and keeps working instead of alarming all night. A face enrolment must
never pretend: it refuses with a CompreFace-shaped 503, because a
silently dropped write is worse than any error. Detection is the
store-acquisition failure the code was already hitting — no polling, no
lock, no second client.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from fastapi import Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from siteloom.config import SiteConfig
from siteloom.store import AuditLog, Identity

log = logging.getLogger(__name__)

#: Hard cap on one upload. Enforced on the bytes actually read; a
#: Content-Length over the cap is turned away earlier as an optimization
#: only, since that header is client-supplied and absent when chunked.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

WINDOW_S = 60.0

#: Retry hint (seconds) on every store-busy answer. The store is
#: typically held for the length of a job, not a blip; this keeps a
#: polling client's next look cheap without inviting a hammering loop.
RETRY_AFTER_S = 30

#: Additive marker on a degraded /recognize body: subjects are empty
#: because nothing was SEARCHED, not because nobody matched. A client
#: that only reads "result" (Double Take does) never sees it.
STORE_BUSY_MARKER = "vector-store-busy"

#: CompreFace's CommonExceptionCode.FACES_SERVICE_EXCEPTION — its own
#: "the face service failed" code, the nearest analogue its public
#: source offers for a backend that cannot serve right now.
FACES_SERVICE_EXCEPTION = 41


class VectorStoreHeld(Exception):
    """Another process holds the embedded vector store's flock.

    Raised by RecognitionService._acquire_vectors so each route answers
    per the CLD-110 decision: reads degrade to a marked no-match, vector
    writes refuse with a CompreFace-shaped 503.
    """


def _recognize_response(result: list[dict], store_busy: bool) -> JSONResponse:
    """The /recognize answer, degraded or not (CLD-110).

    Degraded is CompreFace's own no-match shape — a client that only
    reads "result" (Double Take does) treats it as "nobody recognised"
    and keeps working — plus an additive marker and Retry-After so one
    that cares can tell "nobody matched" from "nobody was looked for".
    The available-store answer is byte-identical to what it always was:
    nothing additive rides along when nothing is degraded.
    """
    if store_busy:
        return JSONResponse(
            {"result": result, "degraded": STORE_BUSY_MARKER},
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )
    return JSONResponse({"result": result})


def _store_busy_refusal() -> JSONResponse:
    """A 503 in CompreFace's error vocabulary, for a write the held
    store makes impossible.

    The body matches CompreFace's ExceptionResponseDto ({"message",
    "code"}, java/common .../dto/ExceptionResponseDto.java) with
    FACES_SERVICE_EXCEPTION as the code. CompreFace pairs that code
    with a 500; answering 503 + Retry-After is the one deliberate
    divergence, because a held store is temporary and retryable, not
    broken. A Double Take-style client surfaces `message` whenever
    `result` is absent (special-casing only code 28, no-faces-found),
    so this reads as an actionable error — never as a silent success
    that dropped an enrolment.
    """
    return JSONResponse(
        {
            "message": (
                "the vector store is held by another process (ingest, "
                "backfill or an enrolment sweep); the face was NOT "
                "enrolled — retry after it finishes"
            ),
            "code": FACES_SERVICE_EXCEPTION,
        },
        status_code=503,
        headers={"Retry-After": str(RETRY_AFTER_S)},
    )


class RateLimiter:
    """Sliding-window request budget per client IP.

    Lock-serialized because FastAPI serves sync endpoints from a
    threadpool (same reasoning as VectorStore's locking).

    Memory is bounded by the sweep, not by the per-IP eviction: draining
    a deque only helps for an IP that comes back, and the traffic a rate
    limiter attracts is precisely the kind that rotates source addresses
    and never returns. Without the sweep the defence against abuse
    becomes the mechanism of a slow OOM, so the tracked set is capped at
    the addresses actually seen in the last window or two.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = 0.0
        self._lock = threading.Lock()

    def allow(self, ip: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        cutoff = now - WINDOW_S
        with self._lock:
            if now - self._last_sweep >= WINDOW_S:
                self._sweep(cutoff)
                self._last_sweep = now
            hits = self._hits.setdefault(ip, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True

    def _sweep(self, cutoff: float) -> None:
        """Drop every address whose most recent hit fell out of the
        window. Caller holds the lock."""
        for ip in [
            ip for ip, hits in self._hits.items() if not hits or hits[-1] <= cutoff
        ]:
            del self._hits[ip]

    @property
    def tracked_ips(self) -> int:
        with self._lock:
            return len(self._hits)


def _declared_too_large(request: Request) -> bool:
    """True when the client announced a body over the cap.

    Only a hint: absent on chunked requests and trivially wrong on a
    lying one, which is why it never replaces the byte count below.
    """
    raw = request.headers.get("content-length")
    if raw is None:
        return False
    try:
        return int(raw) > MAX_UPLOAD_BYTES
    except ValueError:
        return False


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, aborting with 413 once it exceeds the cap.

    This is the real enforcement, and it is deliberately late: FastAPI
    resolves an UploadFile parameter by awaiting request.form(), so
    python-multipart has already parsed — and past its spool threshold,
    written to a temp file — the entire body before this function reads
    its first chunk. The middleware's Content-Length check turns away
    honest oversized clients before that spool; a chunked or lying
    client still gets its whole body received and spooled, and only then
    answered with 413. Bandwidth and disk are therefore bounded only by
    the header check, memory by this one.
    """
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "upload larger than 10 MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RecognitionService:
    """Face detect + embed + match/enroll against the shared stores."""

    def __init__(self, config: SiteConfig, session_factory, vectors=None, embedder=None):
        self.config = config
        self.Session = session_factory
        self._vectors = vectors
        self._embedder = embedder
        self._store_was_busy = False

    @property
    def vectors(self):
        if self._vectors is None:
            from siteloom.identity import get_shared_store

            # Shared process-wide client — embedded Qdrant rejects a
            # second one on the same path (identity/vectors.py).
            self._vectors = get_shared_store(self.config.identity.vector_db_path)
        return self._vectors

    def _acquire_vectors(self):
        """The shared store, or VectorStoreHeld while another process
        (siteloom run, a backfill, an enrolment sweep) has the embedded
        database's flock.

        The busy state is detected from the acquisition failure itself —
        the RuntimeError embedded Qdrant raises on a flock collision —
        never by polling or a second client. Only successes are cached
        (the property leaves _vectors unset on failure), so acquisition
        is retried per request and the API recovers the moment the
        holding process exits, without a restart. Logging is on the
        state TRANSITION: Double Take polls for hours while ingest runs,
        and a warning per frame is alarm spam, which is what this change
        exists to stop.
        """
        try:
            store = self.vectors
        except RuntimeError as exc:
            if not self._store_was_busy:
                self._store_was_busy = True
                log.warning(
                    "vector store is held by another process; recognition "
                    "reads degrade to no-match and enrolments refuse with "
                    "503 until it frees (CLD-110): %s",
                    exc,
                )
            raise VectorStoreHeld(str(exc)) from exc
        if self._store_was_busy:
            self._store_was_busy = False
            log.info("vector store freed; recognition fully restored")
        return store

    @property
    def embedder(self):
        if self._embedder is None:
            from siteloom.identity.embedders import FaceEmbedder

            self._embedder = FaceEmbedder(
                projection_path=self.config.identity.face_projection_path or None
            )
        return self._embedder

    # -- recognition -----------------------------------------------------

    def recognize(
        self,
        image_bytes: bytes,
        limit: int = 0,
        det_prob_threshold: float | None = None,
        prediction_count: int = 1,
    ) -> tuple[list[dict], bool]:
        """Detect, embed and match faces; returns (results, store_busy).

        When the vector store is held, the read DEGRADES (CLD-110):
        detection and embedding are local and still run, so callers get
        the same boxes a genuine no-match carries — only the subject
        search is skipped, every face reports zero subjects, and
        store_busy tells the route to mark the answer. The store is
        acquired lazily, on the first face that produced an embedding,
        exactly where the code touched it before — a no-face image must
        not start holding the flock it never needed.
        """
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("could not decode image")
        det_threshold = (
            det_prob_threshold
            if det_prob_threshold is not None
            else self.config.integrations.recognition_api.det_prob_threshold
        )
        faces = [f for f in self.embedder.detect(image) if f[-1] >= det_threshold]
        faces.sort(key=lambda f: -f[-1])
        if limit:
            faces = faces[:limit]

        results = []
        store = None
        store_busy = False
        with self.Session() as session:
            for face in faces:
                embedding = self.embedder.embed_face(image, face)
                x, y, w, h = (float(v) for v in face[:4])
                box = {
                    "x_min": int(max(0, x)),
                    "y_min": int(max(0, y)),
                    "x_max": int(x + w),
                    "y_max": int(y + h),
                    "probability": round(float(face[-1]), 5),
                }
                subjects: list[dict] = []
                if embedding is not None and not store_busy:
                    if store is None:
                        try:
                            store = self._acquire_vectors()
                        except VectorStoreHeld:
                            store_busy = True
                if embedding is not None and store is not None:
                    hits = store.search("face", embedding, limit=25)
                    best_by_identity: dict[int, float] = {}
                    for hit in hits:
                        prev = best_by_identity.get(hit.identity_id)
                        if prev is None or hit.score > prev:
                            best_by_identity[hit.identity_id] = hit.score
                    scored: dict[str, float] = {}
                    for identity_id, score in best_by_identity.items():
                        identity = session.get(Identity, identity_id)
                        if identity is None or not identity.label:
                            continue  # unknown bucket is not a "subject"
                        if score > scored.get(identity.label, -1.0):
                            scored[identity.label] = score
                    subjects = [
                        {"subject": name, "similarity": round(score, 5)}
                        for name, score in sorted(
                            scored.items(), key=lambda kv: -kv[1]
                        )[: max(1, prediction_count)]
                    ]
                results.append({"box": box, "subjects": subjects})
        return results, store_busy

    # -- enrollment --------------------------------------------------------

    def enroll(self, subject: str, image_bytes: bytes) -> dict:
        """Add one face example to a subject (creating it if needed) —
        exactly what the photo-backfill loop or an Immich exporter calls
        per face. Same collection live recognition reads.

        A write does not degrade (CLD-110): pretending success while the
        store is held would silently drop the enrolment, which is worse
        than any error the caller can see. The store is acquired first,
        before any decode or row flush, so a refusal (VectorStoreHeld,
        the route's 503) provably leaves the database, the store and the
        audit log exactly as they were.
        """
        vectors = self._acquire_vectors()
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("could not decode image")
        faces = self.embedder.detect(image)
        if not faces:
            raise LookupError("no face found in image")
        best = max(faces, key=lambda f: f[-1])
        embedding = self.embedder.embed_face(image, best)
        if embedding is None:
            raise LookupError("face could not be embedded")

        with self.Session() as session:
            identity = self._subject_identity(session, subject, create=True)
            from siteloom.identity.vectors import SOURCE_API

            # No annotation and no stored crop — an API enrollment posts
            # its image and keeps nothing, so `source` is all the
            # provenance there is to record (CLD-84).
            vectors.add("face", embedding, identity.id, source=SOURCE_API)
            identity.vector_count += 1
            identity.appearance_count += 1
            identity.last_seen = _now()
            session.commit()
            return {"image_id": f"{identity.id}-{identity.vector_count}", "subject": subject}

    def subjects(self) -> list[str]:
        with self.Session() as session:
            rows = session.scalars(
                select(Identity.label)
                .filter(Identity.identifier_key == "face", Identity.label.is_not(None))
                .distinct()
                .order_by(Identity.label)
            ).all()
        return list(rows)

    def add_subject(self, subject: str) -> str:
        with self.Session() as session:
            self._subject_identity(session, subject, create=True)
            session.commit()
        return subject

    def faces(self) -> list[dict]:
        with self.Session() as session:
            rows = session.execute(
                select(Identity.label, func.sum(Identity.vector_count))
                .filter(Identity.identifier_key == "face", Identity.label.is_not(None))
                .group_by(Identity.label)
            ).all()
        return [
            {"subject": label, "examples": int(count or 0)} for label, count in rows
        ]

    def _subject_identity(self, session, subject: str, create: bool) -> Identity:
        subject = subject.strip()
        if not subject:
            raise ValueError("subject name required")
        identity = session.scalar(
            select(Identity).filter_by(identifier_key="face", label=subject)
        )
        if identity is None and create:
            identity = Identity(
                identifier_key="face",
                class_name="person",
                label=subject,
                first_seen=_now(),
                last_seen=_now(),
            )
            session.add(identity)
            session.flush()
        if identity is None:
            raise LookupError(f"subject {subject!r} not found")
        return identity


def register(app, config: SiteConfig, service: RecognitionService) -> None:
    api_cfg = config.integrations.recognition_api
    if not api_cfg.api_key and not api_cfg.allow_open:
        # Fail fast at mount time: the operator finds out when the
        # process comes up, not when the first anonymous request
        # enrolls a face into the gallery.
        raise ValueError(
            "integrations.recognition_api is enabled without an api_key. "
            "Set api_key (the 'key' CompreFace clients send as x-api-key), "
            "or set allow_open: true to deliberately serve face matching "
            "and enrollment without authentication."
        )
    if not api_cfg.api_key:
        log.warning(
            "recognition API is serving WITHOUT authentication "
            "(allow_open=true): anyone who can reach this port can match "
            "faces and enroll new ones"
        )

    limiter = RateLimiter(api_cfg.rate_limit_per_minute)
    #: Audit rows need a who with no User row to join (the /api/v1/
    #: prefix is exempt from the console middleware) — same denormalized
    #: username discipline as web/auth.py, one consistent actor string.
    actor = "(api-key)" if api_cfg.api_key else "(api-open)"

    @app.middleware("http")
    async def api_limits(request: Request, call_next):
        """Rate limit and declared-size rejection for /api/v1/ only.

        Both live here rather than in the route dependency because
        FastAPI awaits request.form() before it resolves dependencies:
        anything checked in a Depends has already cost a full body
        parse. Scoped to the API prefix so /healthz, /readyz and /login
        can never be throttled — a probe that gets a 429 gets the
        service killed.
        """
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)
        # Rate-limit before the key check (below) so key brute-forcing
        # is throttled along with everything else.
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        if _declared_too_large(request):
            return JSONResponse(
                {"detail": "upload larger than 10 MB"}, status_code=413
            )
        return await call_next(request)

    def guard(request: Request) -> None:
        supplied = request.headers.get("x-api-key", "")
        if api_cfg.api_key and not hmac.compare_digest(supplied, api_cfg.api_key):
            raise HTTPException(401, "invalid or missing x-api-key")

    def audit(method: str, path: str, status_code: int) -> None:
        with service.Session() as session:
            session.add(
                AuditLog(
                    at=_now(),
                    user_id=None,
                    username=actor,
                    method=method,
                    path=path,
                    status_code=status_code,
                )
            )
            session.commit()

    @app.post("/api/v1/recognition/recognize")
    async def recognize(
        request: Request,
        file: UploadFile,
        limit: int = Query(0),
        det_prob_threshold: float | None = Query(None),
        prediction_count: int = Query(1),
        _=Depends(guard),
    ):
        try:
            result, store_busy = service.recognize(
                await _read_upload(file),
                limit=limit,
                det_prob_threshold=det_prob_threshold,
                prediction_count=prediction_count,
            )
        except (ValueError, LookupError) as exc:
            raise HTTPException(400, str(exc))
        return _recognize_response(result, store_busy)

    @app.get("/api/v1/recognition/subjects")
    def list_subjects(_=Depends(guard)):
        return JSONResponse({"subjects": service.subjects()})

    @app.post("/api/v1/recognition/subjects")
    async def create_subject(request: Request, _=Depends(guard)):
        body = await request.json()
        subject = str(body.get("subject", "")).strip()
        if not subject:
            raise HTTPException(400, "subject name required")
        result = service.add_subject(subject)
        audit("POST", "/api/v1/recognition/subjects", 201)
        return JSONResponse({"subject": result}, status_code=201)

    @app.post("/api/v1/recognition/faces")
    async def add_face(
        file: UploadFile, subject: str = Query(...), _=Depends(guard)
    ):
        try:
            result = service.enroll(subject, await _read_upload(file))
        except VectorStoreHeld:
            # Writes refuse (CLD-110). Not audited — nothing happened,
            # same rule denied requests follow (web/auth.py).
            return _store_busy_refusal()
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except LookupError as exc:
            # CompreFace returns 400 for no-face-found as well.
            raise HTTPException(400, str(exc))
        audit("POST", "/api/v1/recognition/faces", 201)
        return JSONResponse(result, status_code=201)

    @app.get("/api/v1/recognition/faces")
    def list_faces(_=Depends(guard)):
        return JSONResponse({"faces": service.faces()})
