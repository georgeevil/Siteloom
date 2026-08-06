"""Ingest pipeline test with a stub detection module — no YOLO weights
needed; the real model is exercised in the end-to-end smoke run."""

from __future__ import annotations

import pytest

from siteloom.config import (
    CameraConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.dispatch import LocalBackend
from siteloom.ingest import IngestService
from siteloom.store import Detection, Event, EventIdentity, Identity


class StubDetector:
    """Reports one 'person' detection per frame under a fixed track id."""

    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 50.0, 90.0],
                    "track_id": 7,
                    "zones": [],
                    "crop_jpeg": b"\xff\xd8fakejpg",
                }
            ]
        }


class StubIdentity:
    """Returns a constant person embedding, so every frame resolves to
    the same identity."""

    def process(self, job):
        return {
            "embeddings": [
                {
                    "identifier": "person",
                    "algo": "generic",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "plate": None,
                }
            ]
        }


@pytest.fixture
def service(sample_video, tmp_path):
    config = SiteConfig(
        site_id="test-site",
        cameras=[
            CameraConfig(
                id="cam1",
                adapter="file",
                source=str(sample_video),
                sample_fps=5.0,
                modules=["detection"],
            )
        ],
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/test.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    return IngestService(config, dispatcher=dispatcher)


def test_ingest_end_to_end(service):
    count = service.run_camera(service.config.cameras[0])
    assert count == 10  # 30 frames @15fps sampled at 5fps

    with service.Session() as session:
        events = session.query(Event).all()
        detections = session.query(Detection).all()

    # Same track id + class on one camera → one event, many detections.
    assert len(events) == 1
    event = events[0]
    assert event.class_name == "person"
    assert event.track_id == 7
    assert event.detection_count == 10
    assert event.best_crop_path is not None
    assert len(detections) == 10
    assert event.first_seen <= event.last_seen


def test_ingest_respects_max_frames(service):
    assert service.run_camera(service.config.cameras[0], max_frames=3) == 3


def test_ingest_skips_module_not_configured(service, sample_video):
    cam = service.config.cameras[0]
    cam.modules = []  # detection disabled for this camera (NFR3)
    service.run_camera(cam)
    with service.Session() as session:
        assert session.query(Detection).count() == 0


def test_ingest_with_identity_pipeline(sample_video, tmp_path):
    """Full chain with stubs: detection -> identity job -> resolver ->
    Identity + EventIdentity rows and vectors in the local Qdrant."""
    config = SiteConfig(
        site_id="test-site",
        cameras=[
            CameraConfig(
                id="cam1",
                adapter="file",
                source=str(sample_video),
                sample_fps=5.0,
                modules=["detection", "identity"],
            )
        ],
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/test.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    dispatcher.register("identity", StubIdentity())
    service = IngestService(config, dispatcher=dispatcher)
    service.run_camera(config.cameras[0])

    with service.Session() as session:
        identities = session.query(Identity).all()
        links = session.query(EventIdentity).all()

    # Constant embedding -> one identity, re-matched every frame.
    assert len(identities) == 1
    identity = identities[0]
    assert identity.identifier_key == "person"
    assert identity.label is None  # unknown until labeled
    assert identity.appearance_count == 10
    # One event (single track) -> one link, hit-counted per frame.
    assert len(links) == 1
    assert links[0].hit_count == 10
