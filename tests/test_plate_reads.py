"""Plate OCR keeps what it read (CLD-85).

`PlateReader.read()` used to return `str | None`, which meant the only
thing that survived a read was whether it happened to clear a hard-coded
four-character floor. The detector's box confidence picked a region and
was dropped, no OCR confidence was captured, `normalize_plate` threw
away every non-[A-Z0-9] character irreversibly, and a short read — the
motorcycle case CLD-9 is about — returned None leaving no evidence that
a read had been attempted at all.

These tests hold the three things that fix has to get right:

* the rejections are recorded, with their reason and their raw text;
* the plate crop is a *third* image and `crop_jpeg` is untouched by it
  ("one crop, two jobs" — changing that image invalidates every stored
  vector);
* whatever crosses from the identity module to ingest stays serializable,
  because under a Celery/Ray backend it crosses a process boundary.

Nothing here loads a model: the plates dependency group is optional and
is not installed, so the reader, the detector and the OCR are all stubs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from siteloom.config import (
    CameraConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.dispatch import LocalBackend
from siteloom.identity.plates import (
    DEFAULT_MIN_CHARS,
    REASON_LOW_CONFIDENCE,
    REASON_NO_BOX,
    REASON_NO_TEXT,
    REASON_TOO_BLURRY,
    REASON_TOO_SHORT,
    REASON_TOO_SMALL,
    PlateRead as PlateReadResult,
    PlateReader,
    _build_detector,
    gate_reason,
    laplacian_variance,
    mean_confidence,
    min_confidence,
    normalize_plate,
    parse_ocr_result,
    quiet_empty_nms,
)
from siteloom.ingest import IngestService
from siteloom.modules.identity import IdentityModule
from siteloom.store import Detection, Event, PlateRead, get_session, init_db, make_engine
from siteloom.web.app import create_app

TS = datetime(2026, 8, 9, 21, 30)


# --------------------------------------------------------------------------
# Stubs standing in for the optional `plates` dependencies.
# --------------------------------------------------------------------------


class _Box:
    def __init__(self, x1=2, y1=2, x2=30, y2=12):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class _Detection:
    def __init__(self, confidence, box=None):
        self.confidence = confidence
        self.bounding_box = box or _Box()


class FakeDetector:
    """open_image_models' LicensePlateDetector, minus the model."""

    def __init__(self, detections):
        self.detections = detections

    def predict(self, image):
        return list(self.detections)


class Prediction:
    """fast-plate-ocr >= 1.1's PlatePrediction shape."""

    def __init__(self, plate, char_probs=None):
        self.plate = plate
        self.char_probs = char_probs


class FakeOcr:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def run(self, image, **kwargs):
        self.kwargs = kwargs
        return self.result


def reader_with(detections, ocr_result):
    """A PlateReader whose two models are stubs — no weights, no download."""
    reader = object.__new__(PlateReader)
    reader._detector = FakeDetector(detections)
    reader._ocr = FakeOcr(ocr_result)
    return reader


def vehicle_crop():
    return np.full((40, 60, 3), 90, dtype=np.uint8)


def textured_crop(width=400, height=120):
    """A crop with hard vertical edges — a stand-in for legible glyphs.

    A flat image has a Laplacian variance of exactly 0, which is what
    `vehicle_crop()` gives and what makes it the "blurred" fixture; this
    is its opposite, so a sharpness floor can be shown to pass as well
    as to reject.
    """
    crop = np.full((height, width, 3), 20, dtype=np.uint8)
    crop[:, ::4] = 240
    return crop


# --------------------------------------------------------------------------
# The read itself
# --------------------------------------------------------------------------


def test_a_short_read_is_rejected_but_recorded_with_its_reason():
    """The exact case CLD-9 asks about: a plate under the floor.

    It must not resolve — the floor is there to keep junk out of the
    identity store — and it must not vanish either."""
    reader = reader_with([_Detection(0.91)], [Prediction("A1")])
    read = reader.read(vehicle_crop())

    assert read.text is None  # nothing is handed to the resolver
    assert read.accepted is False
    assert read.reason == REASON_TOO_SHORT
    assert read.raw_text == "A1"
    assert read.normalized == "A1"
    assert read.min_chars == DEFAULT_MIN_CHARS


def test_the_floor_is_configuration_so_lowering_it_accepts_the_same_read():
    """Answering CLD-9 is "move the floor and re-read", not a code change."""
    reader = reader_with([_Detection(0.91)], [Prediction("A1")])
    assert reader.read(vehicle_crop(), min_chars=4).text is None
    lowered = reader.read(vehicle_crop(), min_chars=2)
    assert lowered.text == "A1"
    assert lowered.reason is None
    assert lowered.min_chars == 2


def test_raw_text_survives_what_normalization_throws_away():
    """`normalize_plate` is lossy and irreversible, which is what made a
    near-miss indistinguishable from a clean read."""
    reader = reader_with([_Detection(0.8)], [Prediction("kd 1-23 x")])
    read = reader.read(vehicle_crop())

    assert read.raw_text == "kd 1-23 x"
    assert read.normalized == "KD123X" == read.text
    assert normalize_plate(read.raw_text) == read.normalized
    # The whole point: the spacing and punctuation are still readable.
    assert read.raw_text != read.normalized


def test_the_detector_confidence_is_kept_not_just_used_to_pick_a_box():
    reader = reader_with(
        [_Detection(0.42), _Detection(0.87), _Detection(0.13)], [Prediction("ABC123")]
    )
    read = reader.read(vehicle_crop())
    # The highest-confidence box still wins; its score is now recorded.
    assert read.detector_confidence == pytest.approx(0.87)


def test_a_read_with_no_detected_box_is_still_a_read():
    reader = reader_with([], [Prediction("ABC123")])
    read = reader.read(vehicle_crop())

    assert read.text is None
    assert read.reason == REASON_NO_BOX
    assert read.detector_confidence is None
    assert read.plate_jpeg is None


def test_ocr_returning_nothing_is_recorded_as_no_text():
    reader = reader_with([_Detection(0.6)], [])
    read = reader.read(vehicle_crop())
    assert read.reason == REASON_NO_TEXT
    assert read.detector_confidence == pytest.approx(0.6)
    # The crop still exists — seeing what the OCR failed on is the point.
    assert read.plate_jpeg is not None


def test_ocr_confidence_is_captured_when_the_library_reports_one():
    reader = reader_with(
        [_Detection(0.6)], [Prediction("ABC123", char_probs=[0.9, 0.8, 1.0, 0.9])]
    )
    read = reader.read(vehicle_crop())
    assert read.ocr_confidence == pytest.approx(0.9)
    # It is asked for explicitly; a library that does not offer it just
    # reports nothing (below), and nothing is not zero.
    assert reader._ocr.kwargs == {"return_confidence": True}


def test_an_unreported_ocr_confidence_is_absent_never_zero():
    reader = reader_with([_Detection(0.6)], ["ABC123"])
    read = reader.read(vehicle_crop())
    assert read.text == "ABC123"
    assert read.ocr_confidence is None


def test_both_shapes_the_dependency_range_allows_are_read():
    """fast-plate-ocr 1.0 returns strings (optionally with an array of
    per-character probabilities); 1.1 returns PlatePrediction objects.
    The pin is 1.1 and the range allows 1.0, so both are parsed — the old
    `str(texts)` fallback would have normalized a repr into a plate.

    Probabilities come back unreduced: the caller wants the mean and the
    minimum, and a parser that reduced here could return only one.
    """
    assert parse_ocr_result(["AB12"]) == ("AB12", None)
    text, probs = parse_ocr_result((["AB12"], np.array([[0.5, 0.7]])))
    assert (text, mean_confidence(probs)) == ("AB12", pytest.approx(0.6))
    text, probs = parse_ocr_result([Prediction("AB12", char_probs=[1.0, 0.5])])
    assert (text, mean_confidence(probs)) == ("AB12", pytest.approx(0.75))
    # Anything else is "no text", never a stringified object.
    assert parse_ocr_result([object()]) == (None, None)


def test_the_mean_hides_the_character_the_minimum_shows():
    """The whole reason `ocr_min_confidence` exists: one substituted
    character disappears into an average and is the entire failure."""
    probs = [0.98, 0.98, 0.98, 0.98, 0.98, 0.35]
    assert mean_confidence(probs) == pytest.approx(0.875, abs=1e-3)
    assert min_confidence(probs) == pytest.approx(0.35)
    # And an unreported confidence stays absent through both reductions.
    assert mean_confidence(None) is None
    assert min_confidence(None) is None
    assert min_confidence([]) is None


# --------------------------------------------------------------------------
# Image-quality floors — why "confident and wrong" happens, and the fix
# --------------------------------------------------------------------------


def test_every_read_is_measured_even_with_no_floor_set():
    """The floors cannot be chosen without the numbers, so the numbers are
    taken unconditionally — including on a read that clears everything."""
    reader = reader_with([_Detection(0.9)], [Prediction("ABC123", [0.9, 0.5])])
    read = reader.read(vehicle_crop())

    assert read.text == "ABC123"  # no floor set, nothing rejected
    # The box is (2,2)-(30,12) of the vehicle crop.
    assert (read.plate_width, read.plate_height) == (28, 10)
    assert read.sharpness == pytest.approx(0.0)  # a flat crop has no edges
    assert read.ocr_confidence == pytest.approx(0.7)
    assert read.ocr_min_confidence == pytest.approx(0.5)


def test_a_tiny_plate_is_rejected_however_confident_the_ocr_is():
    """The observed failure mode: box confidence 0.92, OCR confidence
    0.86, and six characters read six different ways off a 60-pixel
    plate. No confidence floor catches that, because the model is
    genuinely confident about characters it interpolated — the size of
    the picture is what says the answer could not have been there."""
    reader = reader_with([_Detection(0.92)], [Prediction("Z52576", [0.86] * 6)])
    read = reader.read(vehicle_crop(), min_width=100)

    assert read.text is None
    assert read.reason == REASON_TOO_SMALL
    assert read.plate_width == 28
    # Recorded in full, so lowering the floor is a re-query of the table.
    assert read.normalized == "Z52576"
    assert read.ocr_confidence == pytest.approx(0.86)


def test_a_blurred_plate_is_rejected_and_a_crisp_one_of_the_same_size_is_not():
    """Size alone would pass a large, motion-smeared plate; sharpness is
    what separates the two."""
    box = _Box(0, 0, 400, 120)
    blurred = reader_with([_Detection(0.9, box)], [Prediction("ABC123")])
    assert blurred.read(vehicle_crop(), min_sharpness=50).reason == REASON_TOO_BLURRY

    crisp = reader_with([_Detection(0.9, box)], [Prediction("ABC123")])
    read = crisp.read(textured_crop(), min_sharpness=50)
    assert read.text == "ABC123"
    assert read.sharpness > 50


def test_the_confidence_floor_is_on_the_weakest_character_not_the_mean():
    """Five characters at 0.98 and one at 0.35 average to 0.87 — over any
    mean floor an operator would set, and wrong in exactly one place."""
    probs = [0.98, 0.98, 0.98, 0.98, 0.98, 0.35]
    reader = reader_with([_Detection(0.9)], [Prediction("ABC123", probs)])
    read = reader.read(vehicle_crop(), min_char_confidence=0.5)

    assert read.text is None
    assert read.reason == REASON_LOW_CONFIDENCE
    assert read.ocr_confidence == pytest.approx(0.875, abs=1e-3)
    assert read.ocr_min_confidence == pytest.approx(0.35)


def test_a_confidence_the_ocr_never_reported_cannot_fail_the_floor():
    """Absent is absent, not zero. An OCR build that reports no
    per-character probabilities must not have 100% of its reads rejected
    the moment someone sets a confidence floor."""
    reader = reader_with([_Detection(0.9)], ["ABC123"])
    read = reader.read(vehicle_crop(), min_char_confidence=0.9)

    assert read.text == "ABC123"
    assert read.ocr_min_confidence is None


def test_the_floors_report_the_fault_an_operator_can_act_on():
    """A 30-pixel plate that also read two characters is `too-small`: that
    is the fact worth acting on, and the character floor is not what went
    wrong. Order is diagnosis, so it is pinned."""
    common = dict(min_chars=4, min_width=100, min_sharpness=10, min_char_confidence=0.5)
    assert (
        gate_reason(normalized="AB", plate_width=30, sharpness=2, char_confidence=0.1, **common)
        == REASON_TOO_SMALL
    )
    assert (
        gate_reason(normalized="AB", plate_width=200, sharpness=2, char_confidence=0.1, **common)
        == REASON_TOO_BLURRY
    )
    assert (
        gate_reason(normalized="AB", plate_width=200, sharpness=90, char_confidence=0.1, **common)
        == REASON_TOO_SHORT
    )
    assert (
        gate_reason(normalized="ABCD", plate_width=200, sharpness=90, char_confidence=0.1, **common)
        == REASON_LOW_CONFIDENCE
    )
    assert (
        gate_reason(normalized="ABCD", plate_width=200, sharpness=90, char_confidence=0.9, **common)
        is None
    )


def test_an_unset_floor_is_off_not_a_floor_of_zero():
    """Every floor defaults to 0, and a read must be unaffected by one —
    otherwise adding these columns would change every existing install."""
    assert (
        gate_reason(
            normalized="ABCD",
            plate_width=1,
            sharpness=0.0,
            char_confidence=0.0,
            min_chars=4,
            min_width=0,
            min_sharpness=0.0,
            min_char_confidence=0.0,
        )
        is None
    )


def test_sharpness_of_an_empty_image_is_absent_not_perfectly_blurred():
    assert laplacian_variance(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert laplacian_variance(np.full((8, 8, 3), 5, dtype=np.uint8)) == pytest.approx(0.0)


def test_the_measurements_reach_the_store(sample_video, tmp_path):
    """The module measures, ingest writes — the compute/state split means
    neither half proves this on its own, and a metric that stops at the
    process boundary is a column of NULLs on the screen that needs it."""
    read = PlateReadResult(
        text=None,
        raw_text="Z52576",
        normalized="Z52576",
        detector_confidence=0.92,
        ocr_confidence=0.86,
        ocr_min_confidence=0.41,
        plate_width=58,
        plate_height=19,
        sharpness=12.5,
        reason=REASON_TOO_SMALL,
    ).as_payload()
    service = plate_service(sample_video, tmp_path, read)
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        row = session.query(PlateRead).first()
    assert row is not None
    assert row.plate_width == 58
    assert row.plate_height == 19
    assert row.sharpness == pytest.approx(12.5)
    assert row.ocr_min_confidence == pytest.approx(0.41)
    assert row.reason == REASON_TOO_SMALL


def test_the_floors_travel_from_config_to_the_reader():
    """A floor that lives in config and never reaches `read()` is a
    setting that silently does nothing."""
    import cv2

    from siteloom.dispatch.base import Job

    seen = {}

    class RecordingReader:
        def read(self, crop, **kwargs):
            seen.update(kwargs)
            return PlateReadResult(text=None, reason=REASON_NO_BOX)

    cfg = IdentityConfig()
    ident = cfg.identifiers["vehicle"]
    ident.plate_min_width_px = 110
    ident.plate_min_sharpness = 45.5
    ident.plate_min_char_confidence = 0.6
    ok, jpeg = cv2.imencode(".jpg", vehicle_crop())
    assert ok
    identity_module(RecordingReader(), cfg).process(
        Job(module="identity", payload={"crop_jpeg": jpeg.tobytes(), "class_name": "car"})
    )

    assert seen == {
        "min_chars": ident.plate_min_chars,
        "min_width": 110,
        "min_sharpness": 45.5,
        "min_char_confidence": 0.6,
    }


# --------------------------------------------------------------------------
# The module boundary: compute only, and serializable
# --------------------------------------------------------------------------


class StubEmbedder:
    def embed(self, crop):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def identity_module(reader, cfg=None):
    module = IdentityModule(cfg or IdentityConfig(), device="cpu")
    module.registry.embedder_for = lambda key: StubEmbedder()  # no weights
    module._plate_reader_tried = True
    module._plate_reader = reader
    return module


def _assert_no_arrays(value, path="result"):
    """Nothing in a module result may be an ndarray or a live handle."""
    assert not isinstance(value, np.ndarray), f"{path} is an ndarray"
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_arrays(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _assert_no_arrays(item, f"{path}[{i}]")
    else:
        assert isinstance(
            value, (str, bytes, int, float, bool, type(None))
        ), f"{path} is a {type(value).__name__}"


def test_the_module_result_carries_the_read_and_stays_serializable(tmp_path):
    """The plate sub-crop travels as JPEG bytes. It cannot be an ndarray:
    this dict crosses a process boundary under Celery/Ray."""
    import cv2

    from siteloom.dispatch.base import Job

    module = identity_module(
        reader_with([_Detection(0.77)], [Prediction("XY 99 12", char_probs=[0.9])])
    )
    ok, jpeg = cv2.imencode(".jpg", vehicle_crop())
    assert ok
    result = module.process(
        Job(module="identity", payload={"crop_jpeg": jpeg.tobytes(),
                                        "class_name": "motorcycle"})
    )

    _assert_no_arrays(result)
    vehicle = [e for e in result["embeddings"] if e["identifier"] == "vehicle"]
    assert len(vehicle) == 1
    read = vehicle[0]["plate_read"]
    assert vehicle[0]["plate"] == "XY9912"
    assert read["raw_text"] == "XY 99 12"
    assert read["detector_confidence"] == pytest.approx(0.77)
    assert isinstance(read["plate_jpeg"], bytes)


def test_a_module_with_no_plate_reader_is_unchanged(tmp_path):
    """The plates group is optional; without it the vehicle path degrades
    to visual re-ID and must behave exactly as it did."""
    import cv2

    from siteloom.dispatch.base import Job

    module = identity_module(None)
    module._plate_reader = None
    ok, jpeg = cv2.imencode(".jpg", vehicle_crop())
    result = module.process(
        Job(module="identity", payload={"crop_jpeg": jpeg.tobytes(),
                                        "class_name": "car"})
    )
    vehicle = [e for e in result["embeddings"] if e["identifier"] == "vehicle"][0]
    assert vehicle["plate"] is None
    assert vehicle["plate_read"] is None


# --------------------------------------------------------------------------
# Ingest writes the row (the module never does)
# --------------------------------------------------------------------------


CROP_JPEG = b"\xff\xd8vehicle-crop-bytes"
PLATE_JPEG = b"\xff\xd8plate-crop-bytes"


class StubMotorcycleDetector:
    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "motorcycle",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 90.0, 130.0],
                    "track_id": 3,
                    "zones": [],
                    "crop_jpeg": CROP_JPEG,
                }
            ]
        }


class StubPlateIdentity:
    """An identity module that already ran its OCR — the payload shape
    `modules/identity.py` produces, with no models behind it."""

    def __init__(self, read, vector=(1.0, 0.0, 0.0, 0.0)):
        self.read = read
        self.vector = list(vector) if vector is not None else None

    def process(self, job):
        return {
            "embeddings": [
                {
                    "identifier": "vehicle",
                    "algo": "generic",
                    "vector": self.vector,
                    "plate": self.read["text"],
                    "plate_read": self.read,
                }
            ]
        }


def plate_service(sample_video, tmp_path, read, vector=(1.0, 0.0, 0.0, 0.0)):
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
            db_url=f"sqlite:///{tmp_path}/plates.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubMotorcycleDetector())
    dispatcher.register("identity", StubPlateIdentity(read, vector))
    return IngestService(config, dispatcher=dispatcher)


def rejected_read():
    return PlateReadResult(
        text=None,
        raw_text="a1",
        normalized="A1",
        detector_confidence=0.64,
        ocr_confidence=0.31,
        plate_jpeg=PLATE_JPEG,
        reason=REASON_TOO_SHORT,
        min_chars=4,
    ).as_payload()


def test_a_rejected_read_becomes_a_row_with_its_reason_and_raw_text(
    sample_video, tmp_path
):
    service = plate_service(sample_video, tmp_path, rejected_read())
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        reads = session.query(PlateRead).all()
        # Nothing was resolvable, so no identity was invented from it.
        assert session.query(Event).count() >= 1

    assert reads, "a failed read must still be recorded"
    row = reads[0]
    assert row.accepted is False
    assert row.reason == REASON_TOO_SHORT
    assert row.raw_text == "a1"
    assert row.text == "A1"  # the normalized form is kept as well
    assert row.min_chars == 4
    assert row.detector_confidence == pytest.approx(0.64)
    assert row.ocr_confidence == pytest.approx(0.31)
    assert row.class_name == "motorcycle"
    assert row.camera_id == "cam1"
    assert row.identifier_key == "vehicle"
    assert row.event_id is not None
    assert row.detection_id is not None
    assert row.verdict is None


def test_a_read_with_no_box_is_recorded_too(sample_video, tmp_path):
    read = PlateReadResult(text=None, reason=REASON_NO_BOX).as_payload()
    service = plate_service(sample_video, tmp_path, read)
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        rows = session.query(PlateRead).all()
    assert rows
    assert all(r.reason == REASON_NO_BOX for r in rows)
    assert all(r.raw_text is None and r.crop_path is None for r in rows)
    # A missing confidence reads as missing, never as zero.
    assert all(r.detector_confidence is None for r in rows)


def test_the_plate_crop_is_a_third_image_and_crop_jpeg_is_untouched(
    sample_video, tmp_path
):
    """"One crop, two jobs": `crop_jpeg` is the display thumbnail *and*
    the embedder input, so changing it invalidates every stored vector.
    The evidence image for an OCR read is written beside it."""
    service = plate_service(sample_video, tmp_path, rejected_read())
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        detections = session.query(Detection).all()
        reads = session.query(PlateRead).all()

    detection_crops = {d.crop_path for d in detections}
    assert detection_crops and None not in detection_crops
    # Byte-identical to what the detector emitted: nothing in this change
    # rewrites, re-encodes or re-crops the image the embedders see.
    for path in detection_crops:
        assert Path(path).read_bytes() == CROP_JPEG

    plate_crops = {r.crop_path for r in reads}
    assert plate_crops and None not in plate_crops
    assert plate_crops.isdisjoint(detection_crops)
    for path in plate_crops:
        assert Path(path).read_bytes() == PLATE_JPEG
        assert Path(path).parent.name == "plates"
    # And the row still points back at the vehicle crop it was cut from.
    assert all(r.source_crop_path in detection_crops for r in reads)


def test_an_accepted_read_still_matches_by_plate(sample_video, tmp_path):
    """Instrumenting the read must not change what it does: a plate still
    reaches the resolver and still writes `Identity.plate` (write-once)."""
    from siteloom.store import Identity

    read = PlateReadResult(
        text="ABC123",
        raw_text="ABC-123",
        normalized="ABC123",
        detector_confidence=0.95,
        plate_jpeg=PLATE_JPEG,
        min_chars=4,
    ).as_payload()
    service = plate_service(sample_video, tmp_path, read)
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        identities = session.query(Identity).all()
        rows = session.query(PlateRead).all()
    assert len(identities) == 1
    assert identities[0].plate == "ABC123"
    assert rows and all(r.accepted for r in rows)
    assert rows[0].raw_text == "ABC-123"


# --------------------------------------------------------------------------
# The review screen
# --------------------------------------------------------------------------


def seeded_client(tmp_path, rows, configure=None):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/web.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    if configure is not None:
        configure(config)
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        event = Event(
            camera_id="cam1", class_name="motorcycle", first_seen=TS, last_seen=TS
        )
        session.add(event)
        session.flush()
        for i, row in enumerate(rows):
            session.add(
                PlateRead(
                    event_id=event.id,
                    camera_id="cam1",
                    at=TS + timedelta(seconds=i),
                    identifier_key="vehicle",
                    **row,
                )
            )
        session.commit()
    return TestClient(create_app(config)), Session


def test_the_class_filter_isolates_motorcycles(tmp_path):
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "CAR999", "text": "CAR999",
             "accepted": True},
            {"class_name": "motorcycle", "raw_text": "zz1", "text": "ZZ1",
             "accepted": False, "reason": REASON_TOO_SHORT, "min_chars": 4},
        ],
    )
    # The per-frame log, every status — the rejections no longer lead
    # the default view (CLD-131), but they are all one filter away.
    everything = client.get("/plates?view=reads&status=all").text
    assert "CAR999" in everything and "zz1" in everything

    bikes = client.get("/plates?view=reads&status=all&class=motorcycle").text
    assert "zz1" in bikes
    assert "CAR999" not in bikes
    # The rejection explains itself rather than showing an empty cell.
    assert "under the character floor" in bikes


def test_the_page_shows_what_the_floors_are_measured_against(tmp_path):
    """A floor is chosen from these columns, so they have to be on the
    screen — and the one that explains "confident and still wrong" is the
    plate's size in pixels, not either confidence."""
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "Z52576", "text": "Z52576",
             "accepted": False, "reason": REASON_TOO_SMALL,
             "detector_confidence": 0.92, "ocr_confidence": 0.86,
             "ocr_min_confidence": 0.41, "plate_width": 58,
             "plate_height": 19, "sharpness": 12.4},
        ],
    )
    page = client.get("/plates?view=reads&status=all").text

    assert "58" in page and "19" in page  # the plate's size in pixels
    assert "sharpness 12" in page
    assert "min 0.41" in page  # the weakest character, beside the mean
    assert "plate region too small to read" in page


def test_a_configured_floor_names_the_setting_that_moves_it(tmp_path):
    """A floor an operator cannot see is one they will debug as a bug."""

    def configure(config):
        config.identity.identifiers["vehicle"].plate_min_width_px = 110

    client, _ = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
          "accepted": True}],
        configure=configure,
    )
    page = client.get("/plates").text

    assert "plate_min_width_px" in page
    assert "110px" in page
    # Unset floors stay off the line rather than rendering as "&ge; 0".
    assert "plate_min_sharpness" not in page
    # The character floor is always in force, so it is always named.
    assert "plate_min_chars" in page


def test_the_status_filter_separates_rejections_from_accepted_reads(tmp_path):
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "CAR999", "text": "CAR999",
             "accepted": True},
            {"class_name": "motorcycle", "raw_text": "zz1", "text": "ZZ1",
             "accepted": False, "reason": REASON_TOO_SHORT},
        ],
    )
    rejected = client.get("/plates?view=reads&status=rejected").text
    assert "zz1" in rejected and "CAR999" not in rejected
    accepted = client.get("/plates?view=reads&status=accepted").text
    assert "CAR999" in accepted and "zz1" not in accepted


def test_a_verdict_persists_and_is_reversible(tmp_path):
    client, Session = seeded_client(
        tmp_path,
        [{"class_name": "motorcycle", "raw_text": "AB12", "text": "AB12",
          "accepted": True}],
    )
    with Session() as session:
        read_id = session.query(PlateRead).one().id

    client.post(f"/plates/{read_id}/verdict",
                data={"verdict": "wrong", "back": "/plates?class=motorcycle"})
    with Session() as session:
        row = session.query(PlateRead).one()
        assert row.verdict == "wrong"
        assert row.verdict_at is not None

    # Judging is a filter of its own, so 20 rows can be worked to zero.
    assert "AB12" in client.get("/plates?status=wrong").text
    assert "AB12" not in client.get("/plates?status=unjudged").text

    # Clicking the same button again clears it — a misclick is not a
    # permanent record of the wrong judgement.
    client.post(f"/plates/{read_id}/verdict", data={"verdict": ""})
    with Session() as session:
        row = session.query(PlateRead).one()
        assert row.verdict is None
        assert row.verdict_at is None


def test_an_unknown_verdict_is_refused(tmp_path):
    client, Session = seeded_client(
        tmp_path, [{"class_name": "car", "raw_text": "AB12", "text": "AB12"}]
    )
    with Session() as session:
        read_id = session.query(PlateRead).one().id
    assert client.post(
        f"/plates/{read_id}/verdict", data={"verdict": "maybe"}
    ).status_code == 400


# --------------------------------------------------------------------------
# Search, and the one-plate history page
# --------------------------------------------------------------------------


def test_search_strips_what_plate_matching_strips(tmp_path):
    """"ab-12" finds the row whose raw text was "AB 12-CD": the operator's
    punctuation goes through the same `normalize_plate` the OCR's did."""
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AB 12-CD", "text": "AB12CD",
             "accepted": True},
            {"class_name": "car", "raw_text": "ZZ999", "text": "ZZ999",
             "accepted": True},
        ],
    )
    found = client.get("/plates?view=reads&q=ab-12").text
    assert "AB 12-CD" in found
    assert "ZZ999" not in found


def test_a_search_that_normalizes_to_nothing_matches_nothing(tmp_path):
    """An explicit search must never silently become "no filter"."""
    client, _ = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB12", "text": "AB12",
          "accepted": True}],
    )
    page = client.get("/plates?view=reads&q=--!--").text
    assert "AB12" not in page
    assert "No read matches this filter" in page


def test_search_and_the_class_filter_narrow_together(tmp_path):
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AB12", "text": "AB12",
             "accepted": True},
            {"class_name": "motorcycle", "raw_text": "AB12", "text": "AB12",
             "accepted": True},
        ],
    )
    page = client.get("/plates?q=ab12&class=motorcycle").text
    assert "motorcycle" in page and "AB12" in page
    # The chips keep the search: switching class must not drop q.
    assert "q=ab12" in page


def seeded_plate_page(tmp_path):
    """Two plates, one of them carried by a labeled vehicle identity."""
    from siteloom.store import Identity

    client, Session = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AB 12-CD", "text": "AB12CD",
             "accepted": True},
            {"class_name": "car", "raw_text": "ab12cd", "text": "AB12CD",
             "accepted": True},
            {"class_name": "car", "raw_text": "ZZ999", "text": "ZZ999",
             "accepted": True},
        ],
    )
    with Session() as session:
        session.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                label="the gray van",
                plate="AB12CD",
                first_seen=TS,
                last_seen=TS,
                appearance_count=7,
            )
        )
        session.commit()
        identity_id = session.query(Identity).one().id
    return client, identity_id


def test_the_plate_page_gathers_one_plates_reads_and_its_identity(tmp_path):
    client, identity_id = seeded_plate_page(tmp_path)
    page = client.get("/plates/p/AB12CD").text

    # Both reads of this plate, none of the other one.
    assert page.count("event #") == 2
    assert "ZZ999" not in page
    # The write-once Identity.plate row, linked by name.
    assert "the gray van" in page
    assert f"/identities/{identity_id}" in page


def test_the_plate_page_canonicalizes_its_url(tmp_path):
    client, _ = seeded_plate_page(tmp_path)
    hop = client.get("/plates/p/ab-12cd", follow_redirects=False)
    assert hop.status_code == 307
    assert hop.headers["location"] == "/plates/p/AB12CD"
    assert client.get("/plates/p/--!--").status_code == 404


def test_a_plate_nobody_read_says_so_instead_of_erroring(tmp_path):
    client, _ = seeded_plate_page(tmp_path)
    page = client.get("/plates/p/NEVER1")
    assert page.status_code == 200
    assert "No read of" in page.text


def test_the_list_links_each_read_to_its_plate_page(tmp_path):
    client, _ = seeded_plate_page(tmp_path)
    assert '/plates/p/AB12CD' in client.get("/plates").text


def test_a_verdict_from_the_plate_page_returns_to_it(tmp_path):
    client, _ = seeded_plate_page(tmp_path)
    with_read = client.get("/plates/p/ZZ999").text
    assert "ZZ999" in with_read
    import re

    read_id = int(re.search(r"/plates/(\d+)/verdict", with_read).group(1))
    hop = client.post(
        f"/plates/{read_id}/verdict",
        data={"verdict": "confirmed", "back": "/plates/p/ZZ999"},
        follow_redirects=False,
    )
    assert hop.status_code == 303
    assert hop.headers["location"] == "/plates/p/ZZ999"


# --------------------------------------------------------------------------
# Corrections: "wrong — it actually says…"
# --------------------------------------------------------------------------


def test_correcting_a_read_records_truth_and_derives_the_verdict(tmp_path):
    """Correcting is judging: the verdict follows from whether the typed
    truth agrees with what the OCR read, so a correction and a
    "confirmed" verdict on a misread can never coexist."""
    client, Session = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "ABI2CD", "text": "ABI2CD",
          "accepted": True}],
    )
    with Session() as session:
        read_id = session.query(PlateRead).one().id

    # A disagreeing correction is a wrong verdict, normalized like the OCR's.
    client.post(f"/plates/{read_id}/correct", data={"text": "ab-12cd"})
    with Session() as session:
        row = session.query(PlateRead).one()
        assert row.corrected_text == "AB12CD"
        assert row.verdict == "wrong"
        assert row.verdict_at is not None

    # An agreeing correction confirms.
    client.post(f"/plates/{read_id}/correct", data={"text": "abi2cd"})
    with Session() as session:
        row = session.query(PlateRead).one()
        assert row.corrected_text == "ABI2CD"
        assert row.verdict == "confirmed"

    # Clearing removes the correction; the judgement already made stands.
    client.post(f"/plates/{read_id}/correct", data={"text": ""})
    with Session() as session:
        row = session.query(PlateRead).one()
        assert row.corrected_text is None
        assert row.verdict == "confirmed"

    # Junk that normalizes to nothing is refused, not silently cleared.
    assert (
        client.post(f"/plates/{read_id}/correct", data={"text": "!!!"}).status_code
        == 400
    )


def test_a_corrected_read_moves_to_the_true_plates_page(tmp_path):
    """The per-plate page groups by best-known truth: a misread corrected
    to this plate is evidence the vehicle was here; a read corrected
    away no longer belongs to the page of its misreading."""
    client, Session = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "ABI2CD", "text": "ABI2CD",
          "accepted": True}],
    )
    with Session() as session:
        read_id = session.query(PlateRead).one().id
    client.post(f"/plates/{read_id}/correct", data={"text": "AB12CD"})

    true_page = client.get("/plates/p/AB12CD").text
    assert "ABI2CD" in true_page  # the OCR's misreading, shown as such
    assert "corrected onto this page" in true_page
    assert "No read of" in client.get("/plates/p/ABI2CD").text

    # Search stays permissive — either spelling finds the row.
    assert "ABI2CD" in client.get("/plates?view=reads&q=abi2cd").text
    assert "ABI2CD" in client.get("/plates?view=reads&q=ab12cd").text


# --------------------------------------------------------------------------
# The watchlist
# --------------------------------------------------------------------------


def test_watching_a_plate_lists_it_and_badges_its_reads(tmp_path):
    from siteloom.store import PlateWatch

    client, Session = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB 12-CD", "text": "AB12CD",
          "accepted": True}],
    )
    # The plate is normalized on the way in, like everywhere else.
    client.post(
        "/plates/watchlist", data={"plate": "ab-12cd", "label": "banned van"}
    )
    page = client.get("/plates").text
    assert "banned van" in page
    assert "watched" in page  # the badge on the matching read
    assert "1 sighting" in page
    assert "On the watchlist" in client.get("/plates/p/AB12CD").text

    # Watching twice is an update, not an error or a second row.
    client.post(
        "/plates/watchlist", data={"plate": "AB12CD", "label": "expected guest"}
    )
    with Session() as session:
        watch = session.query(PlateWatch).one()
        assert watch.label == "expected guest"
        watch_id = watch.id

    client.post(f"/plates/watchlist/{watch_id}/delete")
    with Session() as session:
        assert session.query(PlateWatch).count() == 0

    assert (
        client.post("/plates/watchlist", data={"plate": "???"}).status_code == 400
    )


class FakeHooks:
    def __init__(self):
        self.fired = []

    def fire(self, event_type, payload):
        self.fired.append((event_type, payload))
        return 1


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, subtopic, payload):
        self.published.append((subtopic, payload))
        return True


def test_a_watched_plate_fires_the_alarm_once_per_event(sample_video, tmp_path):
    """The first accepted read of a watched plate on an event fires the
    webhook and the MQTT message; later frames of the same visit are the
    same alarm and stay quiet."""
    from siteloom.store import PlateWatch

    read = PlateReadResult(
        text="ABC123",
        raw_text="ABC-123",
        normalized="ABC123",
        detector_confidence=0.95,
        plate_jpeg=PLATE_JPEG,
        min_chars=4,
    ).as_payload()
    service = plate_service(sample_video, tmp_path, read)
    with service.Session() as session:
        session.add(
            PlateWatch(plate="ABC123", label="banned", note="", created_at=TS)
        )
        session.commit()
    service.notifier = FakeHooks()
    service.publisher = FakeBus()
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        accepted_reads = (
            session.query(PlateRead).filter(PlateRead.accepted.is_(True)).count()
        )
    assert accepted_reads > 1, "the dedupe needs repeat reads to be exercised"
    # The identity pipeline's own webhooks (identity.unknown for the
    # freshly minted vehicle) still fire; the watch alarm rides alongside
    # them and exactly once.
    hits = [p for t, p in service.notifier.fired if t == "plate.watchlist"]
    assert len(hits) == 1
    payload = hits[0]
    assert payload["plate"] == "ABC123"
    assert payload["watch_label"] == "banned"
    assert payload["event_id"] is not None
    assert [p for s, p in service.publisher.published if s == "watchlist"] == [
        payload
    ]


def test_an_unwatched_plate_fires_nothing(sample_video, tmp_path):
    read = PlateReadResult(
        text="ABC123",
        raw_text="ABC-123",
        normalized="ABC123",
        detector_confidence=0.95,
        plate_jpeg=PLATE_JPEG,
        min_chars=4,
    ).as_payload()
    service = plate_service(sample_video, tmp_path, read)
    service.notifier = FakeHooks()
    service.publisher = FakeBus()
    service.run_camera(service.config.cameras[0])
    assert [t for t, _ in service.notifier.fired if t == "plate.watchlist"] == []
    assert [s for s, _ in service.publisher.published if s == "watchlist"] == []


def test_an_empty_table_says_why_rather_than_reading_as_quiet(tmp_path):
    """The /noise rule: an empty table must not imply no vehicle came
    past when the real answer is that nothing here can produce a row."""
    client, _ = seeded_client(tmp_path, [])
    assert "Identity resolution is off" in client.get("/plates").text


# -- the CoreML empty-NMS complaint (upstream onnxruntime#20372) ------------
#
# The detector's end-to-end NMS cannot run on CoreML when no box
# survives, so a plateless crop logs an ERROR from ONNX Runtime and a
# WARNING from open_image_models — for the correct answer, on ~11% of
# real crops. Measured over 400 of this site's crops, a CPU session
# (which never hits the bug) found a plate in none of the ones CoreML
# errored on, so the severity is what is wrong, not the outcome.


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="open_image_models.detection.core.yolo_v9.inference",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


COREML_EMPTY_NMS = (
    "An error occurred during model inference: [ONNXRuntimeError] : 1 : FAIL : "
    "Non-zero status code returned while running CoreMLExecutionProvider_... node. "
    "Input (/end2end/Add_1_output_0) has a dynamic shape ({-1}) but the runtime "
    "shape ({0}) has zero elements. This is not supported by the CoreML EP."
)


def test_the_empty_nms_complaint_is_dropped_at_normal_levels():
    """Dropped, not relabelled. Demoting the record alone changes
    nothing — the logger's level was cleared when .warning() was called
    and handlers sit at NOTSET, so a DEBUG-labelled line still prints."""
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.INFO)
    try:
        assert quiet_empty_nms(_record(COREML_EMPTY_NMS)) is False
    finally:
        root.setLevel(before)


def test_it_is_still_there_when_the_app_runs_at_debug():
    """Asked of the root logger, because open_image_models pins its own
    at INFO — keying on that one would hide the escape hatch behind an
    incantation instead of `--log-level DEBUG`."""
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.DEBUG)
    try:
        record = _record(COREML_EMPTY_NMS)
        assert quiet_empty_nms(record) is True
        assert record.levelname == "DEBUG"
    finally:
        root.setLevel(before)


def test_a_genuine_inference_failure_still_warns():
    """The same log line reports real failures. Matching on this one's
    own words is what keeps a broken model loud."""
    record = _record(
        "An error occurred during model inference: [ONNXRuntimeError] : 6 : "
        "RUNTIME_EXCEPTION : Exception during initialization: bad allocation"
    )
    assert quiet_empty_nms(record) is True
    assert record.levelno == logging.WARNING


def test_the_filter_is_installed_once_however_many_readers_exist():
    """One PlateReader per camera is normal; a filter added per reader
    would stack up copies of the same test on every record."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("open_image_models")
    del ort

    detector_log = logging.getLogger(
        "open_image_models.detection.core.yolo_v9.inference"
    )
    before = list(detector_log.filters)
    try:
        _build_detector()
        _build_detector()
        assert detector_log.filters.count(quiet_empty_nms) == 1
    finally:
        detector_log.filters = before


def test_the_detector_session_logs_only_fatal():
    """The native ORT logger writes to stderr, where no Python filter can
    reach it — so it is quieted per session, leaving every other session
    (the OCR model's especially) with all of its diagnostics."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("open_image_models")

    detector_log = logging.getLogger(
        "open_image_models.detection.core.yolo_v9.inference"
    )
    before = list(detector_log.filters)
    try:
        detector = _build_detector()
        options = detector.model.get_session_options()
        assert options.log_severity_level == 4
    finally:
        detector_log.filters = before


# --------------------------------------------------------------------------
# Per-camera plate floors (CLD-128)
# --------------------------------------------------------------------------


def test_plate_floors_resolve_camera_first_then_identifier():
    """One function of the same shape as `threshold_for`: camera override
    field by field, then the identifier's site-wide value — so ingest and
    any future replay cannot disagree about which bar a read faced."""
    from siteloom.config import CameraIdentityOverride, PlateFloorsOverride

    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(
                id="street",
                adapter="file",
                source="x",
                identity=CameraIdentityOverride(
                    plate_floors=PlateFloorsOverride(
                        min_width_px=30, min_char_confidence=0.85
                    )
                ),
            )
        ],
    )
    config.identity.identifiers["vehicle"].plate_min_width_px = 100
    config.identity.identifiers["vehicle"].plate_min_sharpness = 55.0

    floors = config.identity.plate_floors_for("vehicle", config.cameras[0])
    assert floors.min_width_px == 30  # the camera's word wins
    assert floors.min_char_confidence == pytest.approx(0.85)
    assert floors.min_sharpness == pytest.approx(55.0)  # inherited
    assert floors.min_chars == 4  # inherited

    site = config.identity.plate_floors_for("vehicle", None)
    assert site.min_width_px == 100


def test_floors_for_an_unknown_identifier_are_the_bare_defaults():
    config = SiteConfig(
        site_id="t", cameras=[CameraConfig(id="c", adapter="file", source="x")]
    )
    floors = config.identity.plate_floors_for("bike", config.cameras[0])
    assert floors.min_chars == DEFAULT_MIN_CHARS
    assert floors.min_width_px == 0


def test_the_module_prefers_the_floors_in_the_payload():
    """The floors the application layer resolved for the camera reach the
    reader; the identifier's site-wide value is only the fallback."""
    import cv2

    from siteloom.dispatch.base import Job

    cfg = IdentityConfig()
    cfg.identifiers["vehicle"].plate_min_width_px = 100  # would reject 28px
    module = identity_module(
        reader_with([_Detection(0.9)], [Prediction("AB1234", char_probs=[0.99])]),
        cfg,
    )
    ok, jpeg = cv2.imencode(".jpg", vehicle_crop())
    assert ok
    result = module.process(
        Job(
            module="identity",
            payload={
                "crop_jpeg": jpeg.tobytes(),
                "class_name": "car",
                "plate_floors": {"vehicle": {"min_width_px": 20}},
            },
        )
    )
    vehicle = [e for e in result["embeddings"] if e["identifier"] == "vehicle"][0]
    assert vehicle["plate"] == "AB1234"  # 28px clears the camera's 20

    # And without the payload the site-wide 100px floor still bites.
    bare = module.process(
        Job(
            module="identity",
            payload={"crop_jpeg": jpeg.tobytes(), "class_name": "car"},
        )
    )
    vehicle = [e for e in bare["embeddings"] if e["identifier"] == "vehicle"][0]
    assert vehicle["plate"] is None
    assert vehicle["plate_read"]["reason"] == REASON_TOO_SMALL


def test_camera_floor_overrides_are_named_on_the_page(tmp_path):
    """A floor that only bites on one camera is the one an operator will
    debug as a bug — so the page names it (the CLD-128 half of the
    'floors are visible' rule)."""
    from siteloom.config import CameraIdentityOverride, PlateFloorsOverride

    def configure(config):
        config.cameras[0].identity = CameraIdentityOverride(
            plate_floors=PlateFloorsOverride(min_width_px=30)
        )

    client, _ = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
          "accepted": True}],
        configure=configure,
    )
    page = client.get("/plates").text
    assert "cam1: plate width ≥ 30px" in page


# --------------------------------------------------------------------------
# The OCR cadence cap (CLD-130)
# --------------------------------------------------------------------------


class RationedPlateIdentity:
    """A stub that honours the module contract for `skip_plate_ocr`: no
    OCR on a rationed frame, embedding regardless."""

    def __init__(self):
        self.payloads = []

    def process(self, job):
        payload = job.payload
        self.payloads.append(payload)
        skipped = "vehicle" in payload.get("skip_plate_ocr", ())
        read = (
            None
            if skipped
            else PlateReadResult(
                text="AB1234", raw_text="AB1234", normalized="AB1234", min_chars=4
            ).as_payload()
        )
        return {
            "embeddings": [
                {
                    "identifier": "vehicle",
                    "algo": "generic",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "plate": None if read is None else "AB1234",
                    "plate_read": read,
                }
            ]
        }


def _rationed_service(sample_video, tmp_path, module, interval=None):
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
            db_url=f"sqlite:///{tmp_path}/cap.db", media_dir=str(tmp_path / "media")
        ),
    )
    if interval is not None:
        config.identity.identifiers["vehicle"].plate_ocr_interval_s = interval
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubMotorcycleDetector())
    dispatcher.register("identity", module)
    return IngestService(config, dispatcher=dispatcher)


def test_the_ocr_cadence_is_capped_by_frame_time(sample_video, tmp_path):
    """A vehicle dwelling in frame is OCR'd at most once per interval —
    measured in frame time, so backfill and live ration identically —
    while the embedding keeps running on every identified frame."""
    module = RationedPlateIdentity()
    service = _rationed_service(sample_video, tmp_path, module)
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        reads = session.query(PlateRead).order_by(PlateRead.at).all()
        times = [r.at for r in reads]
    assert reads, "the first frame of a visit is always OCR'd"
    # Rationed: strictly fewer OCR attempts than identified frames, and
    # never two attempts inside the interval.
    assert len(module.payloads) > len(reads)
    for earlier, later in zip(times, times[1:]):
        assert (later - earlier).total_seconds() >= 1.0
    # The floors ride in every payload, resolved for the camera.
    assert all("vehicle" in p["plate_floors"] for p in module.payloads)


def test_an_interval_of_zero_reads_every_frame(sample_video, tmp_path):
    """0 = off — the pre-CLD-130 behavior stays one setting away."""
    module = RationedPlateIdentity()
    service = _rationed_service(sample_video, tmp_path, module, interval=0.0)
    service.run_camera(service.config.cameras[0])

    with service.Session() as session:
        count = session.query(PlateRead).count()
    assert count == len(module.payloads)


# --------------------------------------------------------------------------
# The grouped view, the default view, and bulk verdicts (CLD-130/131)
# --------------------------------------------------------------------------


def test_the_default_view_leads_with_accepted_visits(tmp_path):
    """An operator opening /plates is asking "what plates came past?" —
    not for the `no-box` diagnostics and the reads the system already
    refused. Those stay recorded, one chip away, and counted out loud
    (the /noise rule: a filtered table must never read as quiet)."""
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "CAR999", "text": "CAR999",
             "accepted": True},
            {"class_name": "car", "raw_text": "CAR999", "text": "CAR999",
             "accepted": True},
            {"class_name": "car", "raw_text": "CAR999", "text": "CAR999",
             "accepted": True},
            {"class_name": "car", "accepted": False, "reason": REASON_NO_BOX},
            {"class_name": "car", "raw_text": "zz1", "text": "ZZ1",
             "accepted": False, "reason": REASON_TOO_SHORT},
        ],
    )
    page = client.get("/plates").text
    # One row for the visit, not three for the frames.
    assert "CAR999" in page
    assert "3 reads" in page
    # The rejections do not lead — and the no-box diagnostic least of all.
    assert "zz1" not in page
    assert "no plate region found" not in page
    # ...but they are counted, and the escape hatch is offered.
    assert "2 reads in this window are outside" in page
    assert "Every read" in page


def test_a_visit_groups_by_best_known_text(tmp_path):
    """A read corrected onto a plate joins that plate's group: the group
    key is the operator's correction first, the OCR's text second — the
    same best-known truth the per-plate page uses."""
    client, _ = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "TYB506", "text": "TYB506",
             "accepted": True},
            {"class_name": "car", "raw_text": "TYB506", "text": "TYB506",
             "accepted": True},
            {"class_name": "car", "raw_text": "T8B506", "text": "T8B506",
             "accepted": True, "corrected_text": "TYB506"},
        ],
    )
    page = client.get("/plates").text
    assert "3 reads" in page
    assert "T8B506" not in page  # the misreading is inside the group


def test_a_group_expands_to_its_own_reads(tmp_path):
    client, Session = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
             "accepted": True},
            {"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
             "accepted": True},
        ],
    )
    with Session() as session:
        event_id = session.query(Event).one().id
    page = client.get("/plates").text
    assert "view=reads" in page
    assert f"event={event_id}" in page

    reads = client.get(
        f"/plates?view=reads&status=all&event={event_id}&q=AB1234"
    ).text
    assert reads.count("AB1234") >= 2


def test_a_group_verdict_judges_the_visit_not_frame_743(tmp_path):
    client, Session = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
             "accepted": True},
            {"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
             "accepted": True},
            {"class_name": "car", "raw_text": "ZZ999", "text": "ZZ999",
             "accepted": True},
        ],
    )
    with Session() as session:
        event_id = session.query(Event).one().id

    client.post(
        "/plates/bulk",
        data={"action": "confirmed", "group": f"{event_id}:AB1234"},
    )
    with Session() as session:
        judged = session.query(PlateRead).filter_by(verdict="confirmed").all()
        untouched = session.query(PlateRead).filter_by(text="ZZ999").one()
    assert len(judged) == 2
    assert all(r.text == "AB1234" for r in judged)
    assert untouched.verdict is None

    # Clearing a group undoes the verdicts without touching corrections.
    client.post(
        "/plates/bulk",
        data={"action": "clear", "group": f"{event_id}:AB1234"},
    )
    with Session() as session:
        assert session.query(PlateRead).filter(
            PlateRead.verdict.is_not(None)
        ).count() == 0


def test_bulk_verdicts_over_a_selection(tmp_path):
    client, Session = seeded_client(
        tmp_path,
        [
            {"class_name": "car", "raw_text": "AA1111", "text": "AA1111",
             "accepted": True},
            {"class_name": "car", "raw_text": "BB2222", "text": "BB2222",
             "accepted": True},
            {"class_name": "car", "raw_text": "CC3333", "text": "CC3333",
             "accepted": True},
        ],
    )
    with Session() as session:
        ids = [r.id for r in session.query(PlateRead).order_by(PlateRead.id)]

    client.post("/plates/bulk", data={"action": "wrong", "read": ids[:2]})
    with Session() as session:
        wrong = {r.id for r in session.query(PlateRead).filter_by(verdict="wrong")}
    assert wrong == set(ids[:2])

    assert client.post(
        "/plates/bulk", data={"action": "maybe", "read": ids}
    ).status_code == 400
    assert client.post(
        "/plates/bulk", data={"action": "wrong", "group": "notanevent:X"}
    ).status_code == 400


# --------------------------------------------------------------------------
# The timeframe (CLD-131, the picker from CLD-121/115)
# --------------------------------------------------------------------------


def test_the_timeframe_is_a_living_window(tmp_path):
    """Rows seeded at a fixed past instant: a short preset excludes them,
    a long one includes them, and both are judged at request time."""
    client, _ = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB1234", "text": "AB1234",
          "accepted": True}],
    )
    assert "AB1234" not in client.get("/plates?last=1h").text
    assert "AB1234" in client.get("/plates?last=30d").text
    # An unknown token degrades to all time — the picker is the only
    # thing that mints presets.
    assert "AB1234" in client.get("/plates?last=3w").text
    # A hand-mangled absolute bound is a 400, never silently "no filter".
    assert client.get("/plates?since=notatime").status_code == 400
