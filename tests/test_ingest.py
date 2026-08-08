"""Ingest pipeline test with a stub detection module — no YOLO weights
needed; the real model is exercised in the end-to-end smoke run."""

from __future__ import annotations

import pytest

from siteloom.config import (
    CameraConfig,
    EventConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.dispatch import LocalBackend
from siteloom.ingest import IngestService, _bbox_iou
from siteloom.store import Detection, Event, EventIdentity, Identity


class StubDetector:
    """Reports one 'person' detection per frame under a fixed track id.

    The bbox is comfortably above EventConfig.identify_min_crop_px so the
    identity path is exercised, not gated.
    """

    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 80.0, 120.0],
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


class SequenceDetector:
    """Replays a scripted list of per-frame detection lists."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def process(self, job):
        dets = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return {"detections": [dict(d) for d in dets]}


def _det(track_id=None, bbox=(10.0, 10.0, 80.0, 120.0), class_name="person",
         confidence=0.9):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": list(bbox),
        "track_id": track_id,
        "zones": [],
        "crop_jpeg": b"\xff\xd8fakejpg",
    }


def _sequence_service(sample_video, tmp_path, frames, events_cfg=None):
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
        events=events_cfg or EventConfig(),
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/test.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", SequenceDetector(frames))
    return IngestService(config, dispatcher=dispatcher)


def test_bbox_iou():
    assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert _bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_trackless_burst_stitches_to_one_event(sample_video, tmp_path):
    """Detections with no track id and overlapping boxes continue one
    event instead of spawning one event per detection (the noise burst)."""
    service = _sequence_service(
        sample_video, tmp_path, [[_det(track_id=None)] for _ in range(6)]
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
    assert len(events) == 1
    assert events[0].detection_count == 6


def test_fresh_track_ids_are_adopted(sample_video, tmp_path):
    """A tracker rebuild hands out a new id mid-visit; the overlapping box
    stitches it onto the same event and the event adopts the new id."""
    service = _sequence_service(
        sample_video,
        tmp_path,
        [[_det(track_id=1)], [_det(track_id=1)], [_det(track_id=9)], [_det(track_id=9)]],
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
    assert len(events) == 1
    assert events[0].track_id == 9
    assert events[0].detection_count == 4


def test_non_overlapping_detection_starts_a_new_event(sample_video, tmp_path):
    """The IoU guard: same camera+class moments apart but elsewhere in the
    frame is a different subject, not a fragment."""
    service = _sequence_service(
        sample_video,
        tmp_path,
        [
            [_det(track_id=None, bbox=(10, 10, 80, 120))],
            [_det(track_id=None, bbox=(600, 400, 700, 560))],
        ],
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).count() == 2


def test_stitch_tries_multiple_candidates(sample_video, tmp_path):
    """Two trackless subjects alternating frames: the newest event is
    always the *other* subject, so a single-candidate stitcher minted one
    fresh event per frame (CLD-40). Trying the top-k keeps it to two."""
    box_a = (10.0, 10.0, 80.0, 120.0)
    box_b = (600.0, 400.0, 700.0, 560.0)
    frames = []
    for _ in range(3):
        frames.append([_det(track_id=None, bbox=box_a)])
        frames.append([_det(track_id=None, bbox=box_b)])
    service = _sequence_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).order_by(Event.id).all()
    assert len(events) == 2
    assert [e.detection_count for e in events] == [3, 3]


def test_track_link_gap_is_config(sample_video, tmp_path):
    """The track fast-path gap (formerly the hard-coded EVENT_LINK_GAP_S)
    honours config: a gap tighter than the sample interval splits every
    frame of one track into its own event once stitching is off too."""
    service = _sequence_service(
        sample_video,
        tmp_path,
        [[_det(track_id=1)] for _ in range(3)],
        events_cfg=EventConfig(track_link_gap_s=0.1, stitch_gap_s=0.0),
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).count() == 3


def test_mean_confidence_separates_sustained_from_lucky(sample_video, tmp_path):
    """One 0.9 frame among 0.5s: best_confidence says 0.9, the mean says
    what the event actually looked like."""
    frames = [[_det(track_id=1, confidence=0.5)] for _ in range(4)]
    frames.append([_det(track_id=1, confidence=0.9)])
    service = _sequence_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        event = session.query(Event).one()
        assert event.best_confidence == pytest.approx(0.9)
        assert event.mean_confidence == pytest.approx(0.58)


def test_stitching_disabled_by_zero_gap(sample_video, tmp_path):
    service = _sequence_service(
        sample_video,
        tmp_path,
        [[_det(track_id=None)] for _ in range(3)],
        events_cfg=EventConfig(stitch_gap_s=0.0),
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).count() == 3


def test_class_flap_shares_one_event(sample_video, tmp_path):
    """car→truck under one track id is detector flapping, not a new
    vehicle; the group keeps it one event, Detection rows keep the truth."""
    service = _sequence_service(
        sample_video,
        tmp_path,
        [
            [_det(track_id=3, class_name="car")],
            [_det(track_id=3, class_name="truck")],
            [_det(track_id=3, class_name="car")],
        ],
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
        classes = [d.class_name for d in session.query(Detection).order_by(Detection.id)]
    assert len(events) == 1
    assert events[0].class_name == "car"  # keeps its creation class
    assert classes == ["car", "truck", "car"]


def test_significance_flips_at_thresholds_and_sticks(sample_video, tmp_path):
    """Events start ephemeral and earn significance monotonically."""
    two = _sequence_service(
        sample_video, tmp_path, [[_det(track_id=1)] for _ in range(2)]
    )
    two.run_camera(two.config.cameras[0])
    with two.Session() as session:
        assert session.query(Event).one().significant is False

    (tmp_path / "b").mkdir()
    ten = _sequence_service(
        sample_video, tmp_path / "b", [[_det(track_id=1)] for _ in range(10)]
    )
    ten.run_camera(ten.config.cameras[0])
    with ten.Session() as session:
        assert session.query(Event).one().significant is True


def test_low_confidence_never_becomes_significant(sample_video, tmp_path):
    service = _sequence_service(
        sample_video,
        tmp_path,
        [[_det(track_id=1, confidence=0.45)] for _ in range(10)],
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).one().significant is False


def test_per_camera_override_relaxes_the_gate(sample_video, tmp_path):
    from siteloom.config import EventRulesOverride

    service = _sequence_service(
        sample_video, tmp_path, [[_det(track_id=1)]]
    )
    cam = service.config.cameras[0]
    cam.events = EventRulesOverride(min_detections=1)
    service.run_camera(cam)
    with service.Session() as session:
        assert session.query(Event).one().significant is True


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

    # Constant embedding -> one identity, re-matched every frame once the
    # event passes the significance gate (frames 1-2 of 10 are skipped:
    # identity resolution waits for min_detections=3).
    assert len(identities) == 1
    identity = identities[0]
    assert identity.identifier_key == "person"
    assert identity.label is None  # unknown until labeled
    assert identity.appearance_count == 8
    # One event (single track) -> one link, hit-counted per identified frame.
    assert len(links) == 1
    assert links[0].hit_count == 8
    # Match provenance is persisted at ingest — it cannot be
    # reconstructed afterwards (CLD-17's plate-vs-visual split).
    assert links[0].identifier_key == "person"
    assert links[0].matched_by == "visual"
    assert links[0].learned_plate is False


def _identity_service(sample_video, tmp_path, frames, events_cfg=None):
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
        events=events_cfg or EventConfig(),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/test.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", SequenceDetector(frames))
    dispatcher.register("identity", StubIdentity())
    return IngestService(config, dispatcher=dispatcher)


def test_per_camera_identify_gate_overrides_the_site_rule(sample_video, tmp_path):
    """CLD-39: the identify gates are per-camera, not site-wide only.

    Site-wide, identity waits for the event to turn significant (frames
    1-2 of 10 are skipped). A camera that wants everything identified
    says so in its own override — and gets all ten.
    """
    from siteloom.config import EventRulesOverride

    service = _identity_service(
        sample_video, tmp_path, [[_det(track_id=1)] for _ in range(10)]
    )
    cam = service.config.cameras[0]
    cam.events = EventRulesOverride(identify_only_significant=False)
    service.run_camera(cam)
    with service.Session() as session:
        identity = session.query(Identity).one()
        assert identity.appearance_count == 10
    assert service.config.events.identify_only_significant is True  # site untouched


def test_per_camera_identity_threshold_reaches_the_resolver(sample_video, tmp_path):
    """CLD-39: a per-camera threshold is plumbed to the resolver call
    that already takes one — per identifier, on its own scale."""
    from siteloom.config import CameraIdentityOverride

    service = _identity_service(
        sample_video, tmp_path, [[_det(track_id=1)] for _ in range(4)]
    )
    cam = service.config.cameras[0]
    cam.identity = CameraIdentityOverride(thresholds={"person": 0.95})
    seen: list[float | None] = []
    original = service.resolver.resolve

    def spy(*args, **kwargs):
        seen.append(kwargs.get("threshold"))
        return original(*args, **kwargs)

    service.resolver.resolve = spy
    service.run_camera(cam)
    assert seen and set(seen) == {0.95}
    # The identifier's own value is untouched — the camera overrides the
    # call, not the config.
    assert service.config.identity.identifiers["person"].threshold == 0.80


def test_identity_threshold_defaults_to_the_identifier(sample_video, tmp_path):
    service = _identity_service(
        sample_video, tmp_path, [[_det(track_id=1)] for _ in range(4)]
    )
    seen: list[float | None] = []
    original = service.resolver.resolve

    def spy(*args, **kwargs):
        seen.append(kwargs.get("threshold"))
        return original(*args, **kwargs)

    service.resolver.resolve = spy
    service.run_camera(service.config.cameras[0])
    assert seen and set(seen) == {0.80}


def test_same_identity_fragments_merge_into_one_event(sample_video, tmp_path):
    """The subject reappears elsewhere in the frame under a fresh track —
    no box overlap, so stitching can't help — but resolves to the same
    identity moments later: the fragments fold into one event (CLD-40)."""
    box_a = (10.0, 10.0, 80.0, 120.0)
    box_b = (600.0, 400.0, 700.0, 560.0)
    frames = [[_det(track_id=1, bbox=box_a)] for _ in range(4)]
    frames += [[_det(track_id=2, bbox=box_b)] for _ in range(4)]
    service = _identity_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
        links = session.query(EventIdentity).all()
        assert session.query(Identity).count() == 1
    assert len(events) == 1
    assert events[0].detection_count == 8
    assert events[0].track_id == 2  # adopted so later frames fast-path
    assert events[0].mean_confidence == pytest.approx(0.9)
    # One pairing, not one per fragment: hits from frames 3-4 and 7-8.
    assert len(links) == 1
    assert links[0].hit_count == 4


def test_identity_merge_disabled_by_zero_gap(sample_video, tmp_path):
    box_a = (10.0, 10.0, 80.0, 120.0)
    box_b = (600.0, 400.0, 700.0, 560.0)
    frames = [[_det(track_id=1, bbox=box_a)] for _ in range(4)]
    frames += [[_det(track_id=2, bbox=box_b)] for _ in range(4)]
    service = _identity_service(
        sample_video, tmp_path, frames, events_cfg=EventConfig(merge_gap_s=0.0)
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).count() == 2
        assert session.query(EventIdentity).count() == 2
        assert session.query(Identity).count() == 1


def test_ephemeral_fragment_never_resolves_identity(sample_video, tmp_path):
    """A short fragment stays below the significance gate, so identity
    never runs and no unknown-identity row is minted for it."""
    service = _identity_service(
        sample_video, tmp_path, [[_det(track_id=1)] for _ in range(2)]
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Identity).count() == 0
        assert session.query(EventIdentity).count() == 0


def test_weak_or_tiny_detections_skip_identity(sample_video, tmp_path):
    """Low-confidence and small-bbox crops are skipped by the identity
    quality gates even on a significant event."""
    frames = [[_det(track_id=1)] for _ in range(4)]
    frames += [[_det(track_id=1, confidence=0.3)]]  # below identify_min_confidence
    frames += [[_det(track_id=1, bbox=(10.0, 10.0, 40.0, 40.0))]]  # below min crop px
    service = _identity_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        links = session.query(EventIdentity).all()
    # Frames 1-2 pre-gate, 3-4 identify, 5-6 fail the quality gates.
    assert len(links) == 1
    assert links[0].hit_count == 2


def test_ingest_does_not_revive_a_claim_the_operator_unlinked(
    sample_video, tmp_path
):
    """An unlinked claim (CLD-36) is an operator saying this event was
    never that identity. A later matching frame must not quietly restore
    it by incrementing the row's hit count — the correction would vanish
    with no trace that ingest overruled a human. A fresh claim is made
    instead: visible, and correctable in turn.
    """
    from datetime import datetime, timedelta, timezone

    service = _identity_service(sample_video, tmp_path, [])
    cam = service.config.cameras[0]
    base = datetime(2026, 8, 7, 10, 0, 0)
    for i in range(4):  # past min_detections, so identity resolution runs
        service._store_detections(cam, base + timedelta(seconds=i), [_det(track_id=1)])

    with service.Session() as session:
        link = session.query(EventIdentity).one()
        link.unlinked_at = datetime.now(timezone.utc).replace(tzinfo=None)
        link.verdict = "wrong"
        session.commit()
        event_id, identity_id, link_id = link.event_id, link.identity_id, link.id

    # The subject is still in frame and still matches the same identity.
    service._store_detections(cam, base + timedelta(seconds=4), [_det(track_id=1)])

    with service.Session() as session:
        assert session.query(Event).count() == 1  # same visit, one event
        rows = (
            session.query(EventIdentity)
            .filter_by(event_id=event_id, identity_id=identity_id)
            .all()
        )
        detached = next(r for r in rows if r.id == link_id)
        assert detached.unlinked_at is not None
        assert detached.verdict == "wrong"
        fresh = [r for r in rows if r.id != link_id]
        assert len(fresh) == 1
        assert fresh[0].unlinked_at is None and fresh[0].hit_count == 1
