"""CompreFace-compatible API: recognize, enroll, subjects, auth.

Uses a stub face embedder (colour-square faces, fixed embeddings per
colour) so no model downloads are needed and matches are deterministic.
"""

from __future__ import annotations

import io
from datetime import datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.identity import VectorStore
from siteloom.store import Identity, get_session, init_db, make_engine
from siteloom.web.app import create_app
from siteloom.web.recognition_api import RecognitionService

TS = datetime(2026, 8, 5, 12, 0)

COLOURS = {
    "ann": (255, 0, 0),
    "bob": (0, 255, 0),
}
VECTORS = {
    "ann": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "bob": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
}


class StubFace:
    dim = 4

    def detect(self, bgr):
        faces = []
        for colour in COLOURS.values():
            mask = (
                np.abs(bgr.astype(np.int16) - np.array(colour[::-1], dtype=np.int16))
                .max(axis=-1) < 40
            )
            if mask.sum() < 50:
                continue
            ys, xs = np.where(mask)
            row = np.zeros(15, dtype=np.float32)
            row[:4] = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
            row[-1] = 0.95
            faces.append(row)
        return faces

    def embed_face(self, bgr, face):
        x, y, w, h = (int(v) for v in face[:4])
        mean = bgr[y : y + h, x : x + w].reshape(-1, 3).mean(axis=0)[::-1]
        best, dist = None, 1e9
        for name, colour in COLOURS.items():
            d = float(np.abs(mean - np.array(colour, dtype=np.float32)).sum())
            if d < dist:
                best, dist = name, d
        return VECTORS[best]


def photo(*people) -> bytes:
    image = np.full((120, 240, 3), 240, dtype=np.uint8)
    for i, person in enumerate(people):
        x = 20 + i * 110
        image[30:100, x : x + 70] = COLOURS[person][::-1]
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def empty_photo() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((60, 60, 3), 240, dtype=np.uint8))
    return buf.tobytes()


@pytest.fixture
def client_and_session(tmp_path):
    config = SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vec")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/api.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    vectors = VectorStore(config.identity.vector_db_path)
    service = RecognitionService(config, Session, vectors=vectors, embedder=StubFace())
    app = create_app(config, recognition_service=service)
    yield TestClient(app), Session, vectors, config
    vectors.close()


@pytest.fixture
def client(client_and_session):
    return client_and_session[0]


def enroll(client, subject: str, image: bytes):
    return client.post(
        f"/api/v1/recognition/faces?subject={subject}",
        files={"file": ("f.jpg", io.BytesIO(image), "image/jpeg")},
    )


def recognize(client, image: bytes, **params):
    return client.post(
        "/api/v1/recognition/recognize",
        params=params,
        files={"file": ("f.jpg", io.BytesIO(image), "image/jpeg")},
    )


def test_enroll_creates_subject_and_faces(client_and_session):
    client, Session, _v, _c = client_and_session
    r = enroll(client, "Ann Person", photo("ann"))
    assert r.status_code == 201
    assert r.json()["subject"] == "Ann Person"
    # Subject backed by a labeled face Identity — the shared store.
    with Session() as session:
        identity = session.query(Identity).one()
        assert identity.identifier_key == "face"
        assert identity.label == "Ann Person"
        assert identity.vector_count == 1
    assert client.get("/api/v1/recognition/subjects").json() == {
        "subjects": ["Ann Person"]
    }
    assert client.get("/api/v1/recognition/faces").json() == {
        "faces": [{"subject": "Ann Person", "examples": 1}]
    }


def test_recognize_returns_compreface_shape(client):
    enroll(client, "Ann Person", photo("ann"))
    r = recognize(client, photo("ann"))
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result) == 1
    face = result[0]
    assert set(face["box"]) == {"x_min", "y_min", "x_max", "y_max", "probability"}
    assert face["subjects"][0]["subject"] == "Ann Person"
    assert face["subjects"][0]["similarity"] > 0.99


def test_recognize_multiple_faces(client):
    enroll(client, "Ann", photo("ann"))
    enroll(client, "Bob", photo("bob"))
    result = recognize(client, photo("ann", "bob")).json()["result"]
    assert len(result) == 2
    assert {f["subjects"][0]["subject"] for f in result} == {"Ann", "Bob"}


def test_recognize_unknown_face_has_no_subjects(client):
    # Bob is enrolled; Ann's orthogonal embedding scores ~0 similarity —
    # below any sane match, but CompreFace still reports the candidate
    # list, so we assert the top candidate is a poor match, not absent.
    enroll(client, "Bob", photo("bob"))
    result = recognize(client, photo("ann")).json()["result"]
    assert len(result) == 1
    subjects = result[0]["subjects"]
    assert subjects == [] or subjects[0]["similarity"] < 0.1


def test_recognize_no_face_found(client):
    result = recognize(client, empty_photo()).json()["result"]
    assert result == []


def test_unlabeled_identities_are_not_subjects(client_and_session):
    client, Session, vectors, _c = client_and_session
    # An unknown-bucket identity (label=None) with a vector must never
    # be reported as a CompreFace subject.
    with Session() as session:
        identity = Identity(
            identifier_key="face", class_name="person",
            first_seen=TS, last_seen=TS, vector_count=1,
        )
        session.add(identity)
        session.commit()
        vectors.add("face", VECTORS["ann"], identity.id)
    result = recognize(client, photo("ann")).json()["result"]
    assert result[0]["subjects"] == []
    assert client.get("/api/v1/recognition/subjects").json() == {"subjects": []}


def test_create_subject_endpoint(client):
    r = client.post("/api/v1/recognition/subjects", json={"subject": "Cara"})
    assert r.status_code == 201
    assert "Cara" in client.get("/api/v1/recognition/subjects").json()["subjects"]


def test_enroll_without_face_is_400(client):
    r = enroll(client, "Ann", empty_photo())
    assert r.status_code == 400
    assert "no face" in r.json()["detail"]


def test_api_key_enforced(tmp_path):
    config = SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vec2")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/api2.db", media_dir=str(tmp_path / "m2")
        ),
    )
    config.integrations.recognition_api.api_key = "sekret"
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    service = RecognitionService(config, Session, vectors=None, embedder=StubFace())
    client = TestClient(create_app(config, recognition_service=service))

    assert client.get("/api/v1/recognition/subjects").status_code == 401
    assert (
        client.get(
            "/api/v1/recognition/subjects", headers={"x-api-key": "wrong"}
        ).status_code
        == 401
    )
    ok = client.get("/api/v1/recognition/subjects", headers={"x-api-key": "sekret"})
    assert ok.status_code == 200
