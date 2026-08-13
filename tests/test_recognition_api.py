"""CompreFace-compatible API: recognize, enroll, subjects, auth.

Uses a stub face embedder (colour-square faces, fixed embeddings per
colour) so no model downloads are needed and matches are deterministic.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.identity import VectorStore
from siteloom.store import AuditLog, Identity, get_session, init_db, make_engine
from siteloom.web.app import create_app
from siteloom.web.recognition_api import (
    MAX_UPLOAD_BYTES,
    RETRY_AFTER_S,
    RateLimiter,
    RecognitionService,
)

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


def make_config(tmp_path, tag: str = "") -> SiteConfig:
    return SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(vector_db_path=str(tmp_path / f"vec{tag}")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/api{tag}.db",
            media_dir=str(tmp_path / f"media{tag}"),
        ),
    )


@pytest.fixture
def client_and_session(tmp_path):
    config = make_config(tmp_path)
    # The API no longer mounts by default and keyless serving is an
    # explicit opt-in (CLD-47); these tests exercise the open mode.
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.allow_open = True
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


def make_client(config) -> TestClient:
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    service = RecognitionService(config, Session, vectors=None, embedder=StubFace())
    return TestClient(create_app(config, recognition_service=service))


def test_api_key_enforced(tmp_path):
    config = make_config(tmp_path, "2")
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.api_key = "sekret"
    client = make_client(config)

    assert client.get("/api/v1/recognition/subjects").status_code == 401
    assert (
        client.get(
            "/api/v1/recognition/subjects", headers={"x-api-key": "wrong"}
        ).status_code
        == 401
    )
    ok = client.get("/api/v1/recognition/subjects", headers={"x-api-key": "sekret"})
    assert ok.status_code == 200


def test_disabled_by_default_does_not_mount(tmp_path):
    # Biometric surface, off by default (NFR5): a default config must
    # not expose /api/v1/ at all.
    client = make_client(make_config(tmp_path, "3"))
    assert client.get("/api/v1/recognition/subjects").status_code == 404


def test_enabled_keyless_without_opt_in_fails_fast(tmp_path):
    config = make_config(tmp_path, "4")
    config.integrations.recognition_api.enabled = True  # no key, no opt-in
    with pytest.raises(ValueError, match="allow_open"):
        make_client(config)


def test_allow_open_serves_and_warns(tmp_path, caplog):
    config = make_config(tmp_path, "5")
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.allow_open = True
    with caplog.at_level(logging.WARNING, logger="siteloom.web.recognition_api"):
        client = make_client(config)
    assert any("WITHOUT authentication" in r.message for r in caplog.records)
    assert client.get("/api/v1/recognition/subjects").status_code == 200


def test_oversized_upload_is_413(client):
    # 10 MB + 1 byte of not-an-image: the size gate must trip on the
    # bytes actually read, before any decode is attempted.
    blob = b"\0" * (10 * 1024 * 1024 + 1)
    assert enroll(client, "Ann", blob).status_code == 413
    assert recognize(client, blob).status_code == 413


def test_oversized_content_length_rejected_before_the_body(client):
    """A declared oversize body is refused without the multipart parser
    ever running — the only check that happens before the spool."""
    r = client.post(
        "/api/v1/recognition/recognize",
        headers={
            "content-type": "multipart/form-data; boundary=x",
            "content-length": str(MAX_UPLOAD_BYTES + 1),
        },
        content=b"",  # never parsed: the header alone decides
    )
    assert r.status_code == 413
    # A garbage Content-Length must not 500 the route; the byte count
    # stays the real gate.
    ok = client.post(
        "/api/v1/recognition/recognize",
        headers={"content-length": "not-a-number"},
        files={"file": ("f.jpg", io.BytesIO(photo("ann")), "image/jpeg")},
    )
    assert ok.status_code == 200


def test_rate_limiter_does_not_grow_without_bound():
    """An attacker rotating source addresses must not be able to grow
    the tracked set forever — that turns the abuse defence into a slow
    OOM (the CLD-66 bug shape)."""
    limiter = RateLimiter(per_minute=60)
    for i in range(5000):
        assert limiter.allow(f"10.0.{i // 256}.{i % 256}", now=1.0)
    assert limiter.tracked_ips == 5000  # all inside one window, still live

    # One request a full window later: the sweep drops every address
    # whose last hit fell out of the window.
    assert limiter.allow("10.9.9.9", now=1.0 + 61.0)
    assert limiter.tracked_ips == 1

    # Repeated rotation across many windows stays bounded rather than
    # accumulating one entry per address ever seen.
    for window in range(20):
        base = 100.0 + window * 61.0
        for i in range(100):
            limiter.allow(f"192.168.{window}.{i}", now=base)
    assert limiter.tracked_ips <= 200


def test_rate_limiter_still_counts_a_returning_client():
    # The sweep must not become an amnesty: a client inside its window
    # is still held to the budget.
    limiter = RateLimiter(per_minute=3)
    assert [limiter.allow("1.2.3.4", now=1000.0 + i) for i in range(4)] == [
        True,
        True,
        True,
        False,
    ]
    # Once the window has passed, the same client is served again.
    assert limiter.allow("1.2.3.4", now=1000.0 + 61.0) is True


def test_rate_limit_trips_but_probes_survive(tmp_path):
    config = make_config(tmp_path, "6")
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.allow_open = True
    config.integrations.recognition_api.rate_limit_per_minute = 3
    client = make_client(config)

    codes = [
        client.get("/api/v1/recognition/subjects").status_code for _ in range(4)
    ]
    assert codes == [200, 200, 200, 429]
    # The limiter is scoped to /api/v1/ — supervision probes must keep
    # answering while it is tripped (a throttled probe kills the service).
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)
    assert client.get("/readyz").json()  # answered, not throttled


# -- CLD-110: the vector store is held by another process ------------------
#
# The normal condition on a live site: `siteloom run` holds the embedded
# store for hours. Reads degrade to a marked no-match so Double Take
# keeps working; vector writes refuse with a CompreFace-shaped 503 so no
# enrolment is ever silently dropped. Simulated the CLD-62 way — the
# RuntimeError embedded Qdrant raises on a flock collision — never by
# opening a second real client on one path.

LOCK_MESSAGE = "Storage folder vectors is already accessed by another instance"


def make_env(tmp_path, tag: str):
    """App + Session with a service that acquires the store lazily
    (vectors=None), so a monkeypatched get_shared_store decides."""
    config = make_config(tmp_path, tag)
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.allow_open = True
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    service = RecognitionService(config, Session, vectors=None, embedder=StubFace())
    return TestClient(create_app(config, recognition_service=service)), Session, config


@pytest.fixture
def held_env(tmp_path, monkeypatch):
    client, Session, _config = make_env(tmp_path, "held")

    def refuse(*args, **kwargs):
        raise RuntimeError(LOCK_MESSAGE)

    monkeypatch.setattr("siteloom.identity.get_shared_store", refuse)
    return client, Session


def test_recognize_during_held_store_degrades_to_no_match(held_env):
    """Reads degrade: 200 in CompreFace's own no-match shape — the face
    still boxed, zero subjects — with a machine-readable busy marker and
    a Retry-After, so a polling consumer logs "unknown" and keeps
    working instead of alarming through every soak."""
    client, _Session = held_env
    r = recognize(client, photo("ann"))
    assert r.status_code == 200
    assert r.headers["retry-after"] == str(RETRY_AFTER_S)
    body = r.json()
    # Pin the wire marker as a literal: it is a contract with clients,
    # not an implementation detail a rename may drag along.
    assert body["degraded"] == "vector-store-busy"
    # Marker aside, identical in shape to a genuine no-match: same
    # top-level key, same per-face keys, same box keys.
    assert set(body) == {"result", "degraded"}
    (face,) = body["result"]
    assert set(face) == {"box", "subjects"}
    assert set(face["box"]) == {"x_min", "y_min", "x_max", "y_max", "probability"}
    assert face["subjects"] == []


def test_recognize_with_the_store_available_is_unchanged(client):
    """No behaviour change when the store is free: no marker, no
    Retry-After — the exact pre-CLD-110 body, nothing additive."""
    enroll(client, "Ann", photo("ann"))
    r = recognize(client, photo("ann"))
    assert r.status_code == 200
    assert "retry-after" not in r.headers
    body = r.json()
    assert set(body) == {"result"}
    assert body["result"][0]["subjects"][0]["subject"] == "Ann"


def test_face_add_during_held_store_refuses_and_enrolls_nothing(held_env):
    """Writes refuse: a degraded write would silently drop an enrolment
    the caller believes succeeded, which is worse than any error. The
    refusal is CompreFace's ExceptionResponseDto shape ({"message",
    "code"}), code 41 = FACES_SERVICE_EXCEPTION — readable by a client
    written against CompreFace, which surfaces `message` whenever
    `result` is absent."""
    client, Session = held_env
    r = enroll(client, "Ann Person", photo("ann"))
    assert r.status_code == 503
    assert r.headers["retry-after"] == str(RETRY_AFTER_S)
    body = r.json()
    assert set(body) == {"message", "code"}
    assert body["code"] == 41
    assert "NOT enrolled" in body["message"]
    # Nothing happened, provably: no Identity row, no subject, and no
    # audit row (denied requests are not audited — web/auth.py's rule).
    with Session() as session:
        assert session.scalars(select(Identity)).all() == []
        assert session.scalars(select(AuditLog)).all() == []
    assert client.get("/api/v1/recognition/subjects").json() == {"subjects": []}


def test_subject_endpoints_survive_a_held_store(held_env):
    """Subjects live in SQL, not vectors — verified rather than assumed:
    listing and creating them must keep answering while the store is
    held (creation writes no vectors, so it does not refuse)."""
    client, _Session = held_env
    r = client.post("/api/v1/recognition/subjects", json={"subject": "Cara"})
    assert r.status_code == 201
    assert client.get("/api/v1/recognition/subjects").json() == {"subjects": ["Cara"]}
    assert client.get("/api/v1/recognition/faces").json() == {
        "faces": [{"subject": "Cara", "examples": 0}]
    }


def test_recognition_recovers_the_moment_the_store_frees(tmp_path, monkeypatch):
    """A failed acquisition must not be cached: the next request after
    the holding process exits serves matches again, no restart. Run
    against a real store, which also proves the refused write landed
    nowhere — not the database, not the vector store."""
    client, _Session, config = make_env(tmp_path, "recover")
    store = VectorStore(config.identity.vector_db_path)
    held = True

    def gate(path):
        if held:
            raise RuntimeError(LOCK_MESSAGE)
        return store

    monkeypatch.setattr("siteloom.identity.get_shared_store", gate)
    try:
        # Held: the write refuses, the read degrades.
        assert enroll(client, "Ann", photo("ann")).status_code == 503
        assert recognize(client, photo("ann")).json()["degraded"] == "vector-store-busy"

        held = False
        # Freed: the very next read is ordinary — and empty-handed,
        # because the refused enrolment wrote nothing to the store.
        r = recognize(client, photo("ann"))
        assert "retry-after" not in r.headers
        assert set(r.json()) == {"result"}
        assert r.json()["result"][0]["subjects"] == []
        # And writes work again, end to end.
        assert enroll(client, "Ann", photo("ann")).status_code == 201
        r = recognize(client, photo("ann"))
        assert r.json()["result"][0]["subjects"][0]["subject"] == "Ann"
    finally:
        store.close()


def test_api_mutations_are_audited(client_and_session):
    client, Session, _v, _c = client_and_session
    client.post("/api/v1/recognition/subjects", json={"subject": "Cara"})
    enroll(client, "Ann", photo("ann"))
    recognize(client, photo("ann"))  # read path: not a mutation, no row
    with Session() as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.id)).all()
        assert [(r.username, r.path, r.status_code) for r in rows] == [
            ("(api-open)", "/api/v1/recognition/subjects", 201),
            ("(api-open)", "/api/v1/recognition/faces", 201),
        ]
