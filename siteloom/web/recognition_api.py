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
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from siteloom.config import SiteConfig
from siteloom.store import Identity

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RecognitionService:
    """Face detect + embed + match/enroll against the shared stores."""

    def __init__(self, config: SiteConfig, session_factory, vectors=None, embedder=None):
        self.config = config
        self.Session = session_factory
        self._vectors = vectors
        self._embedder = embedder

    @property
    def vectors(self):
        if self._vectors is None:
            from siteloom.identity import get_shared_store

            # Shared process-wide client — embedded Qdrant rejects a
            # second one on the same path (identity/vectors.py).
            self._vectors = get_shared_store(self.config.identity.vector_db_path)
        return self._vectors

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
    ) -> list[dict]:
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
                if embedding is not None:
                    hits = self.vectors.search("face", embedding, limit=25)
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
        return results

    # -- enrollment --------------------------------------------------------

    def enroll(self, subject: str, image_bytes: bytes) -> dict:
        """Add one face example to a subject (creating it if needed) —
        exactly what the photo-backfill loop or an Immich exporter calls
        per face. Same collection live recognition reads."""
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
            self.vectors.add("face", embedding, identity.id)
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

    def check_key(request: Request) -> None:
        if api_cfg.api_key and request.headers.get("x-api-key") != api_cfg.api_key:
            raise HTTPException(401, "invalid or missing x-api-key")

    @app.post("/api/v1/recognition/recognize")
    async def recognize(
        request: Request,
        file: UploadFile,
        limit: int = Query(0),
        det_prob_threshold: float | None = Query(None),
        prediction_count: int = Query(1),
        _=Depends(check_key),
    ):
        try:
            result = service.recognize(
                await file.read(),
                limit=limit,
                det_prob_threshold=det_prob_threshold,
                prediction_count=prediction_count,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return JSONResponse({"result": result})

    @app.get("/api/v1/recognition/subjects")
    def list_subjects(_=Depends(check_key)):
        return JSONResponse({"subjects": service.subjects()})

    @app.post("/api/v1/recognition/subjects")
    async def create_subject(request: Request, _=Depends(check_key)):
        body = await request.json()
        subject = str(body.get("subject", "")).strip()
        if not subject:
            raise HTTPException(400, "subject name required")
        return JSONResponse({"subject": service.add_subject(subject)}, status_code=201)

    @app.post("/api/v1/recognition/faces")
    async def add_face(
        file: UploadFile, subject: str = Query(...), _=Depends(check_key)
    ):
        try:
            result = service.enroll(subject, await file.read())
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except LookupError as exc:
            # CompreFace returns 400 for no-face-found as well.
            raise HTTPException(400, str(exc))
        return JSONResponse(result, status_code=201)

    @app.get("/api/v1/recognition/faces")
    def list_faces(_=Depends(check_key)):
        return JSONResponse({"faces": service.faces()})
