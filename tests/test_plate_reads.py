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
    REASON_NO_BOX,
    REASON_NO_TEXT,
    REASON_TOO_SHORT,
    PlateRead as PlateReadResult,
    PlateReader,
    normalize_plate,
    parse_ocr_result,
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
    `str(texts)` fallback would have normalized a repr into a plate."""
    assert parse_ocr_result(["AB12"]) == ("AB12", None)
    text, conf = parse_ocr_result((["AB12"], np.array([[0.5, 0.7]])))
    assert (text, conf) == ("AB12", pytest.approx(0.6))
    text, conf = parse_ocr_result([Prediction("AB12", char_probs=[1.0, 0.5])])
    assert (text, conf) == ("AB12", pytest.approx(0.75))
    # Anything else is "no text", never a stringified object.
    assert parse_ocr_result([object()]) == (None, None)


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


def seeded_client(tmp_path, rows):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/web.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
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
    everything = client.get("/plates").text
    assert "CAR999" in everything and "zz1" in everything

    bikes = client.get("/plates?class=motorcycle").text
    assert "zz1" in bikes
    assert "CAR999" not in bikes
    # The rejection explains itself rather than showing an empty cell.
    assert "under the character floor" in bikes


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
    rejected = client.get("/plates?status=rejected").text
    assert "zz1" in rejected and "CAR999" not in rejected
    accepted = client.get("/plates?status=accepted").text
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
    found = client.get("/plates?q=ab-12").text
    assert "AB 12-CD" in found
    assert "ZZ999" not in found


def test_a_search_that_normalizes_to_nothing_matches_nothing(tmp_path):
    """An explicit search must never silently become "no filter"."""
    client, _ = seeded_client(
        tmp_path,
        [{"class_name": "car", "raw_text": "AB12", "text": "AB12",
          "accepted": True}],
    )
    page = client.get("/plates?q=--!--").text
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
    assert "ABI2CD" in client.get("/plates?q=abi2cd").text
    assert "ABI2CD" in client.get("/plates?q=ab12cd").text


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
