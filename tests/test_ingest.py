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
         confidence=0.9, crop=b"\xff\xd8fakejpg"):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": list(bbox),
        "track_id": track_id,
        "zones": [],
        "crop_jpeg": crop,
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


# -- occlusion (siteloom/tracking/occlusion.py wiring) ---------------------

BIG_BOX = (400.0, 100.0, 600.0, 500.0)
#: Inside BIG_BOX — the hidden subject's partial box.
SLIVER = (480.0, 320.0, 540.0, 440.0)
FAR_BOX = (900.0, 100.0, 1000.0, 300.0)


def test_occlusion_is_measured_onto_detection_rows(sample_video, tmp_path):
    """Frames sampled while another track's box sat over this one carry
    occluded=True; clear frames carry False, never NULL — NULL is
    reserved for writers with no monitor (the CLD-254 honesty rule)."""
    frames = [
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=2, bbox=FAR_BOX)],
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=2, bbox=SLIVER)],
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=2, bbox=SLIVER)],
    ]
    service = _sequence_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        rows = session.query(Detection).order_by(Detection.id).all()
    # Frame 1: separated — both clear. Frame 2: the overlap begins; the
    # episode cannot confirm on one frame. Frame 3: confirmed — both
    # participants are marked.
    assert [r.occluded for r in rows[:2]] == [False, False]
    assert [r.occluded for r in rows[2:4]] == [False, False]
    assert [r.occluded for r in rows[4:6]] == [True, True]


def test_suspect_birth_is_iou_stitched_by_default(sample_video, tmp_path):
    """The trap, preserved for the before-picture: a sliver track born
    inside an older track's box overlaps *the occluder*, so the IoU
    stitch hands it the occluder's event and the event adopts the
    phantom's track id."""
    frames = [
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=7, bbox=SLIVER)],
    ]
    service = _sequence_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
    assert len(events) == 1
    assert events[0].track_id == 7  # the phantom stole the event


def test_occlusion_stitch_keeps_a_suspect_birth_apart(sample_video, tmp_path):
    """With occlusion_stitch on, the suspect birth skips the IoU stitch
    and starts its own event — position is exactly what lies during an
    occlusion, so only appearance evidence (the identity-aware merge,
    CLD-41-gated) may fold it back."""
    frames = [
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=7, bbox=SLIVER)],
        [_det(track_id=1, bbox=BIG_BOX), _det(track_id=7, bbox=SLIVER)],
    ]
    service = _sequence_service(
        sample_video, tmp_path, frames,
        events_cfg=EventConfig(occlusion_stitch=True),
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).order_by(Event.id).all()
    assert len(events) == 2
    assert events[0].track_id == 1
    assert events[0].detection_count == 4
    assert events[1].track_id == 7
    assert events[1].detection_count == 2


def test_an_ordinary_entrance_still_stitches_with_occlusion_stitch_on(
    sample_video, tmp_path
):
    """The flag must only bite suspects: a fresh track id appearing where
    an event just was (the tracker-rebuild case) still stitches by IoU."""
    frames = [
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=1, bbox=BIG_BOX)],
        [_det(track_id=9, bbox=BIG_BOX)],
    ]
    service = _sequence_service(
        sample_video, tmp_path, frames,
        events_cfg=EventConfig(occlusion_stitch=True),
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).all()
    assert len(events) == 1
    assert events[0].track_id == 9


class AppearanceStub:
    """Identity stub for the swap check: appearance_only probes get a
    vector keyed by the crop's content; ordinary identity jobs resolve
    nothing, keeping the resolver out of the picture."""

    VECTORS = {b"subject-A": [1.0, 0.0], b"subject-B": [0.0, 1.0]}

    def process(self, job):
        if job.payload.get("appearance_only"):
            return {"embeddings": [{
                "identifier": "_appearance", "algo": "generic",
                "vector": self.VECTORS.get(job.payload["crop_jpeg"]),
                "quality": None, "plate": None, "plate_read": None,
            }], "fingerprint": None}
        return {"embeddings": []}


def test_a_mid_occlusion_swap_flags_both_events(sample_video, tmp_path):
    """The failure no other metric sees: both tracks survive the
    occlusion, but their appearances crossed — track 1's later crops
    look like event 2's earlier ones and vice versa. Both events get
    `suspect_swap`, with the scores in the note."""
    apart = (900.0, 100.0, 1000.0, 300.0)
    a = dict(crop=b"subject-A")
    b = dict(crop=b"subject-B")
    frames = [
        # Separated: three pre-episode frames each.
        *[[_det(track_id=1, bbox=BIG_BOX, **a), _det(track_id=2, bbox=apart, **b)]
          for _ in range(3)],
        # Overlap: two frames of the sliver inside track 1's box.
        *[[_det(track_id=1, bbox=BIG_BOX, **a), _det(track_id=2, bbox=SLIVER, **a)]
          for _ in range(2)],
        # Reappearance, appearances crossed: the tracker swapped them.
        *[[_det(track_id=1, bbox=BIG_BOX, **b), _det(track_id=2, bbox=apart, **a)]
          for _ in range(3)],
    ]
    service = _identity_service(
        sample_video, tmp_path, frames, identity_module=AppearanceStub()
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        events = session.query(Event).order_by(Event.id).all()
        assert len(events) == 2
        assert all(e.suspect_swap for e in events)
        import json as _json

        note = _json.loads(events[0].suspect_swap_note)
        assert note["other_event"] == events[1].id
        assert sorted(note["crossed"]) == ["a", "b"]


def test_a_clean_reappearance_flags_nothing(sample_video, tmp_path):
    apart = (900.0, 100.0, 1000.0, 300.0)
    a = dict(crop=b"subject-A")
    b = dict(crop=b"subject-B")
    frames = [
        *[[_det(track_id=1, bbox=BIG_BOX, **a), _det(track_id=2, bbox=apart, **b)]
          for _ in range(3)],
        *[[_det(track_id=1, bbox=BIG_BOX, **a), _det(track_id=2, bbox=SLIVER, **a)]
          for _ in range(2)],
        *[[_det(track_id=1, bbox=BIG_BOX, **a), _det(track_id=2, bbox=apart, **b)]
          for _ in range(3)],
    ]
    service = _identity_service(
        sample_video, tmp_path, frames, identity_module=AppearanceStub()
    )
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert session.query(Event).filter(Event.suspect_swap).count() == 0


def test_cross_class_overlap_is_not_occlusion(sample_video, tmp_path):
    """Containment cannot see depth: a person whose box sits inside a
    parked car's box is standing in FRONT of it. Marking every such
    frame occluded would permanently gate that walkway's learning, so
    only same-class-group tracks can occlude each other."""
    frames = [
        [_det(track_id=1, bbox=BIG_BOX, class_name="car"),
         _det(track_id=2, bbox=SLIVER, class_name="person")]
        for _ in range(4)
    ]
    service = _sequence_service(sample_video, tmp_path, frames)
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        assert all(
            r.occluded is False
            for r in session.query(Detection).all()
        )


def test_a_departing_subject_swap_is_caught_one_sided(sample_video, tmp_path):
    """The most common swap presentation: A leaves during the overlap
    and the tracker keeps A's id on B, so only one track survives the
    episode. The vanished side has no post-episode frames — that must
    not silence the check."""
    from datetime import datetime, timedelta

    apart = (900.0, 100.0, 1000.0, 300.0)
    service = _identity_service(
        sample_video, tmp_path, [], identity_module=AppearanceStub()
    )
    cam = service.config.cameras[0]
    t0 = datetime(2026, 8, 19, 18, 6, 39)

    def at(i):
        return t0 + timedelta(seconds=i * 0.2)

    for i in range(3):  # separated: pre-episode evidence for both
        service._store_detections(cam, at(i), [
            _det(track_id=1, bbox=BIG_BOX, crop=b"subject-A"),
            _det(track_id=2, bbox=apart, crop=b"subject-B"),
        ])
    for i in range(3, 5):  # the overlap
        service._store_detections(cam, at(i), [
            _det(track_id=1, bbox=BIG_BOX, crop=b"subject-A"),
            _det(track_id=2, bbox=BIG_BOX, crop=b"subject-A"),
        ])
    for i in range(5, 20):  # B walks on wearing A's track id; A is gone
        service._store_detections(cam, at(i), [
            _det(track_id=1, bbox=BIG_BOX, crop=b"subject-B"),
        ])
    with service.Session() as session:
        flagged = session.query(Event).filter(Event.suspect_swap).all()
        assert len(flagged) == 2
        import json as _json

        roles = {_json.loads(e.suspect_swap_note)["role"] for e in flagged}
        assert roles == {"a", "b"}


def test_a_suspect_prior_is_never_merged_into(sample_video, tmp_path):
    """The freeze must hold through the identity-aware merge: folding a
    fresh fragment into a frozen event would make a new claim on it —
    the exact thing the freeze forbids — and hand the fragment's clean
    rows to whatever verdict the operator later passes."""
    from datetime import datetime, timedelta

    service = _identity_service(
        sample_video, tmp_path, [],
        events_cfg=EventConfig(min_detections=1, min_duration_s=0.0,
                               min_confidence=0.0),
    )
    cam = service.config.cameras[0]
    t0 = datetime(2026, 8, 19, 18, 6, 39)
    for i in range(3):  # event A, linked to the stub's constant identity
        service._store_detections(
            cam, t0 + timedelta(seconds=i * 0.2), [_det(track_id=1)]
        )
    with service.Session() as session:
        frozen = session.query(Event).one()
        assert session.query(EventIdentity).count() == 1
        frozen.suspect_swap = True
        session.commit()
        frozen_id = frozen.id
    # A new track, far away (no IoU stitch), resolves to the same
    # identity moments later — prime identity-merge bait.
    for i in range(3, 6):
        service._store_detections(
            cam, t0 + timedelta(seconds=i * 0.2),
            [_det(track_id=2, bbox=(600.0, 400.0, 700.0, 560.0))],
        )
    with service.Session() as session:
        events = session.query(Event).order_by(Event.id).all()
        assert [e.id for e in events][0] == frozen_id
        assert len(events) == 2  # not folded into the frozen event
        links = session.query(EventIdentity).all()
        # The fragment claims the identity on its own row; the frozen
        # event gained nothing.
        assert {link.event_id for link in links} == {e.id for e in events}


def test_a_suspected_swap_freezes_identity_claims(sample_video, tmp_path):
    """While the flag stands, the event's frames make no new claims and
    teach nothing — its crops may belong to the other subject."""
    service = _identity_service(
        sample_video, tmp_path, [[_det(track_id=1)]],
        events_cfg=EventConfig(min_detections=1, min_duration_s=0.0,
                               min_confidence=0.0),
    )
    from datetime import datetime

    cam = service.config.cameras[0]
    with service.Session() as session:
        event = Event(camera_id="cam1", track_id=1, class_name="person",
                      first_seen=datetime(2026, 8, 1),
                      last_seen=datetime(2026, 8, 1),
                      significant=True, suspect_swap=True)
        session.add(event)
        session.flush()
        survived = service._identify(
            session, cam, event, _det(track_id=1),
            event.last_seen, None, service.config.events,
        )
        assert survived is event
        assert session.query(EventIdentity).count() == 0


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

    # Constant embedding -> one identity, re-matched every frame. Identity
    # resolution waits for min_detections=3, but the two warm-up frames
    # are identified retroactively when the event flips (CLD-286), so all
    # ten frames count.
    assert len(identities) == 1
    identity = identities[0]
    assert identity.identifier_key == "person"
    assert identity.label is None  # unknown until labeled
    assert identity.appearance_count == 10
    # One event (single track) -> one link, hit-counted per identified frame.
    assert len(links) == 1
    assert links[0].hit_count == 10
    # Match provenance is persisted at ingest — it cannot be
    # reconstructed afterwards (CLD-17's plate-vs-visual split).
    assert links[0].identifier_key == "person"
    assert links[0].matched_by == "visual"
    assert links[0].learned_plate is False


def _identity_service(
    sample_video, tmp_path, frames, events_cfg=None, identity_module=None
):
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
    dispatcher.register("identity", identity_module or StubIdentity())
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
    # One pairing, not one per fragment: each fragment's warm-up frames
    # are identified retroactively at its own flip (CLD-286), so all
    # eight frames hit.
    assert len(links) == 1
    assert links[0].hit_count == 8


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


class RecordingIdentity(StubIdentity):
    """StubIdentity that records the crop bytes of every job it sees."""

    def __init__(self):
        self.crops = []

    def process(self, job):
        self.crops.append(job.payload["crop_jpeg"])
        return super().process(job)


def test_backlog_replays_warmup_frames_in_order_at_the_flip(
    sample_video, tmp_path
):
    """CLD-286: the frames spent earning significance are identified
    retroactively when the event flips — from the stored crops, oldest
    first, before the flipping frame — and exactly once each. A departure's
    biggest plate frames come first, so losing them to warm-up meant losing
    the plate (event 578)."""
    frames = []
    for i in range(5):
        d = _det(track_id=1)
        d["crop_jpeg"] = b"\xff\xd8crop%d" % i
        frames.append([d])
    recorder = RecordingIdentity()
    service = _identity_service(
        sample_video, tmp_path, frames, identity_module=recorder
    )
    service.run_camera(service.config.cameras[0])
    # min_detections=3: the flip lands on frame 3, whose _identify runs
    # after the backlog — so the identity module sees one chronological
    # visit, which is also what keeps plate-OCR rationing honest.
    assert recorder.crops == [b"\xff\xd8crop%d" % i for i in range(5)]
    with service.Session() as session:
        assert session.query(EventIdentity).one().hit_count == 5


def test_backlog_is_capped_when_the_flip_was_delayed(
    sample_video, tmp_path, monkeypatch
):
    """CLD-286: min_detections-1 rows is the usual backlog, but a flip
    held back by min_duration_s (or min_confidence) can sit on far more —
    the replay is capped, keeping the earliest rows (a departure's best
    plate frames come first)."""
    import siteloom.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "IDENTIFY_BACKLOG_MAX", 2)
    frames = [[_det(track_id=1)] for _ in range(10)]
    recorder = RecordingIdentity()
    service = _identity_service(
        sample_video,
        tmp_path,
        frames,
        events_cfg=EventConfig(min_duration_s=1.0),
        identity_module=recorder,
    )
    service.run_camera(service.config.cameras[0])
    # At 5 fps the flip lands on frame 6 (1.0 s after first_seen): five
    # warm-up rows are eligible, the cap keeps the earliest two, and
    # frames 6-10 identify live.
    assert len(recorder.crops) == 7
    with service.Session() as session:
        assert session.query(EventIdentity).one().hit_count == 7


def test_backlog_skips_rows_whose_identity_job_already_ran(
    sample_video, tmp_path
):
    """CLD-286: `events retag` can un-flip an event and a later frame
    re-flip it — `identified_at` marks the rows already resolved, so a
    second backlog pass replays nothing and hit counts stay honest."""
    frames = [[_det(track_id=1)] for _ in range(5)]
    recorder = RecordingIdentity()
    service = _identity_service(
        sample_video, tmp_path, frames, identity_module=recorder
    )
    cam = service.config.cameras[0]
    service.run_camera(cam)
    assert len(recorder.crops) == 5
    with service.Session() as session:
        rows = session.query(Detection).order_by(Detection.timestamp).all()
        assert all(r.identified_at is not None for r in rows)
        event = session.query(Event).one()
        # What a re-flip would run: the pass finds every row stamped.
        service._identify_backlog(session, cam, event, rows[-1], service._rules_for(cam))
    assert len(recorder.crops) == 5


def test_two_trackless_detections_in_one_frame_keep_their_own_crops(
    sample_video, tmp_path
):
    """Two same-class detections with no track id share timestamp, class
    and track ('None') in the crop filename — without the per-frame
    sequence suffix the second overwrites the first, and the backlog pass
    (CLD-286) would then embed the wrong subject's pixels."""
    a = _det(track_id=None, bbox=(10.0, 10.0, 80.0, 120.0))
    a["crop_jpeg"] = b"\xff\xd8subjectA"
    b = _det(track_id=None, bbox=(600.0, 400.0, 700.0, 560.0))
    b["crop_jpeg"] = b"\xff\xd8subjectB"
    service = _sequence_service(sample_video, tmp_path, [[a, b]])
    service.run_camera(service.config.cameras[0])
    with service.Session() as session:
        rows = session.query(Detection).order_by(Detection.id).all()
    assert len(rows) == 2
    assert rows[0].crop_path != rows[1].crop_path
    from pathlib import Path

    assert Path(rows[0].crop_path).read_bytes() == b"\xff\xd8subjectA"
    assert Path(rows[1].crop_path).read_bytes() == b"\xff\xd8subjectB"


def test_backlog_respects_the_per_frame_quality_gates(sample_video, tmp_path):
    """CLD-286: only the significance gate is behind the backlog pass by
    construction — a weak warm-up frame stays out after the flip too."""
    frames = [[_det(track_id=1, confidence=0.3)] for _ in range(2)]
    frames += [[_det(track_id=1)] for _ in range(3)]
    recorder = RecordingIdentity()
    service = _identity_service(
        sample_video, tmp_path, frames, identity_module=recorder
    )
    service.run_camera(service.config.cameras[0])
    # The 0.3 frames sit below identify_min_confidence: the backlog query
    # finds them but _identify's own gates refuse them, same as live.
    assert len(recorder.crops) == 3
    with service.Session() as session:
        assert session.query(EventIdentity).one().hit_count == 3


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
    # Frames 1-2 are recovered by the backlog pass at the flip (CLD-286),
    # 3-4 identify live, 5-6 fail the quality gates either way.
    assert len(links) == 1
    assert links[0].hit_count == 4


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


def test_a_wrong_verdict_stops_the_gallery_growing_mid_visit(sample_video, tmp_path):
    """A wrong match used to keep teaching for the rest of the visit
    (CLD-139): every later frame of the same event added another vector
    to the gallery an operator had just called wrong, so the mistake
    recruited its own reinforcements.

    What stops is learning, and only learning — the event goes on, the
    claim goes on counting its sightings, and the operator's own verdict
    stays where they put it.
    """
    from datetime import datetime, timedelta

    service = _identity_service(sample_video, tmp_path, [])
    cam = service.config.cameras[0]
    # The per-event cap would halt growth on its own after 3, so switch it
    # off: the verdict has to be the thing that stops this.
    service.config.identity.identifiers["person"].learn_max_per_event = 0
    base = datetime(2026, 8, 7, 10, 0, 0)
    for i in range(4):  # past min_detections, so identity resolution runs
        service._store_detections(cam, base + timedelta(seconds=i), [_det(track_id=1)])

    with service.Session() as session:
        identity = session.query(Identity).one()
        link = session.query(EventIdentity).one()
        assert identity.vector_count > 1  # it really was accreting
        vectors_at_verdict, hits_at_verdict = identity.vector_count, link.hit_count
        link.verdict = "wrong"
        link.verdict_at = base
        session.commit()

    # The subject is still in frame, still matching the same identity.
    for i in range(4, 9):
        service._store_detections(cam, base + timedelta(seconds=i), [_det(track_id=1)])

    with service.Session() as session:
        identity = session.query(Identity).one()
        assert identity.vector_count == vectors_at_verdict  # gallery frozen
        link = session.query(EventIdentity).one()
        assert link.hit_count > hits_at_verdict  # the claim still counts frames
        assert link.verdict == "wrong"  # and the judgment is untouched
        assert session.query(Event).count() == 1  # one visit throughout
