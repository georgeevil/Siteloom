"""Vehicle fingerprint (CLD-254): color reads, behind a flag, honestly.

The attribute layer copies the plate-read discipline, and these tests
hold it to the same three things `test_plate_reads.py` holds OCR to:

* a read that names no color is still a measurement with a reason and
  its numbers, so the floors are retuned by reading the table;
* an achromatic crop — the always-IR front-yard camera's every frame —
  reads "no color here", never a confidently wrong "gray";
* the flag is a real off switch: no payload key travels, no columns are
  written, no chip renders.

Nothing here loads a model: color is pure pixel math, the ingest run
uses stub modules, and the module-wiring test rides a class the
registry is configured to ignore so no embedder is ever built.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from siteloom.config import (
    CameraConfig,
    IdentifierConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.dispatch import LocalBackend
from siteloom.dispatch.base import Job
from siteloom.identity.fingerprint import (
    REASON_NO_CHROMA,
    REASON_TOO_SMALL,
    read_color,
    visit_color,
)
from siteloom.ingest import IngestService
from siteloom.modules.identity import IdentityModule
from siteloom.store import (
    Camera,
    Detection,
    Event,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 20, 15, 0, 0)


def solid(bgr: tuple[int, int, int], size: int = 100) -> np.ndarray:
    return np.full((size, size, 3), bgr, dtype=np.uint8)


# --------------------------------------------------------------------------
# read_color: measure first, gate after, record both
# --------------------------------------------------------------------------


def test_a_red_crop_reads_red_with_its_measurements():
    read = read_color(solid((0, 0, 220)), min_px=32, chroma_floor=12.0)
    assert read.color == "red"
    assert read.confidence == pytest.approx(1.0)
    assert read.chroma_p95 == pytest.approx(220.0)
    assert read.reason is None
    # The measurements and the floors applied ride on the read (the
    # CLD-128 rule): the row can be re-judged later without re-running.
    assert read.crop_px == 100
    assert read.min_px == 32
    assert read.chroma_floor == pytest.approx(12.0)


def test_a_grayscale_crop_names_no_color_and_says_why():
    """An IR frame is pure grayscale. Naming its gray 'gray' would be a
    confident guess; the honest answer is 'cannot measure color here'."""
    read = read_color(solid((128, 128, 128)), min_px=32, chroma_floor=12.0)
    assert read.color is None
    assert read.reason == REASON_NO_CHROMA
    # The measurement that failed the floor is recorded, not just the
    # verdict — absent would be indistinguishable from never-measured.
    assert read.chroma_p95 == pytest.approx(0.0)
    assert read.saturation is not None


def test_a_white_car_in_a_chromatic_frame_still_reads_white():
    """What separates a white car from an IR frame is the crop margin:
    `crop_margin` grows the crop past the bbox, so a daylight crop
    carries chromatic background even around an achromatic vehicle."""
    crop = solid((200, 120, 40))  # blue-ish daylight background
    crop[20:80, 20:80] = (255, 255, 255)  # the vehicle fills the center
    read = read_color(crop, min_px=32, chroma_floor=12.0)
    assert read.color == "white"
    assert read.reason is None


def test_a_tiny_crop_is_refused_like_a_tiny_plate():
    read = read_color(solid((0, 0, 220), size=20), min_px=32, chroma_floor=12.0)
    assert read.color is None
    assert read.reason == REASON_TOO_SMALL
    # Refused is still measured: "would lowering min_px recover this
    # camera's reads?" must be answerable from the rows.
    assert read.crop_px == 20
    assert read.min_px == 32
    assert read.chroma_floor == pytest.approx(12.0)


def test_blue_and_black_land_in_their_bins():
    assert read_color(solid((210, 60, 10)), min_px=32, chroma_floor=12.0).color == "blue"
    dark = solid((20, 20, 24))
    assert read_color(dark, min_px=32, chroma_floor=12.0).reason == REASON_NO_CHROMA
    # Black with any chroma present (e.g. from the margin) is named.
    dark[:10, :] = (200, 120, 40)
    assert read_color(dark, min_px=32, chroma_floor=12.0).color == "black"


def test_the_payload_is_plain_scalars():
    """The read crosses a process boundary under a Celery/Ray backend —
    same contract as PlateRead.as_payload."""
    payload = read_color(solid((0, 0, 220)), min_px=32, chroma_floor=12.0).as_payload()
    json.dumps(payload)  # would raise on ndarrays or numpy scalars


# --------------------------------------------------------------------------
# visit_color: display-time grouping, per-frame rows untouched
# --------------------------------------------------------------------------


def test_consensus_is_confidence_weighted_majority():
    reads = [("white", 0.9, None)] * 3 + [("gray", 0.4, None)] * 4
    vc = visit_color(reads)
    assert vc is not None
    assert vc.color == "white"  # 2.7 outweighs 1.6 despite fewer frames
    assert vc.agreeing == 3
    assert vc.named == 7


def test_an_all_ir_visit_reports_the_reason_not_a_guess():
    vc = visit_color([(None, None, REASON_NO_CHROMA)] * 5)
    assert vc is not None
    assert vc.color is None
    assert vc.unnamed_reasons == {REASON_NO_CHROMA: 5}


def test_nothing_measured_is_nothing_not_unknown():
    assert visit_color([]) is None
    assert visit_color([(None, None, None)] * 3) is None


def test_the_badge_blames_the_dominant_reason_not_membership():
    """One grayscale frame in a mostly-too-small visit is a crop-size
    problem — attributing it to IR would assert 'every read saw a
    grayscale crop' about something else entirely."""
    vc = visit_color(
        [(None, None, REASON_TOO_SMALL)] * 4 + [(None, None, REASON_NO_CHROMA)]
    )
    assert vc is not None
    assert vc.dominant_unnamed_reason == REASON_TOO_SMALL


def test_the_template_speaks_the_module_reason_vocabulary():
    """The chip branches on reason strings; renaming a constant must
    fail here, not silently degrade the badge."""
    from pathlib import Path

    import siteloom.web

    template = (
        Path(siteloom.web.__file__).parent / "templates" / "event.html"
    ).read_text()
    assert f"== '{REASON_NO_CHROMA}'" in template
    assert f"== '{REASON_TOO_SMALL}'" in template


# --------------------------------------------------------------------------
# One resolution decides the gate for ingest and display alike
# --------------------------------------------------------------------------


def test_fingerprint_request_follows_the_vehicle_identifier_by_default():
    cfg = IdentityConfig()
    cfg.fingerprint.enabled = True
    assert cfg.fingerprint_request("car") == {"min_px": 32, "chroma_floor": 12.0}
    assert cfg.fingerprint_request("person") is None
    # Adding a class to the vehicle identifier — CLAUDE.md's one-step
    # way to re-identify it — fingerprints it too; no second list to
    # keep in sync.
    cfg.identifiers["vehicle"].applies_to.append("van")
    assert cfg.fingerprint_request("van") is not None
    # A named list pins the set independently.
    cfg.fingerprint.classes = ["truck"]
    assert cfg.fingerprint_request("van") is None
    assert cfg.fingerprint_request("truck") is not None


def test_fingerprint_request_is_none_when_off_or_without_vehicles():
    assert IdentityConfig().fingerprint_request("car") is None
    cfg = IdentityConfig(identifiers={"face": IdentifierConfig(
        algo="face", applies_to=["person"])})
    cfg.fingerprint.enabled = True
    assert cfg.fingerprint_request("car") is None


# --------------------------------------------------------------------------
# Module wiring: computed only when asked, travels even when empty
# --------------------------------------------------------------------------


def crop_jpeg(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_the_module_reads_color_only_when_the_payload_asks():
    # "bird" is in auto_add_exclude, so no identifier applies and no
    # embedder (hence no model) is ever built — the fingerprint read is
    # per crop, not per identifier, and must run anyway.
    module = IdentityModule(IdentityConfig(), device="cpu")
    payload = {"crop_jpeg": crop_jpeg(solid((0, 0, 220))), "class_name": "bird"}

    silent = module.process(Job(module="identity", payload=payload))
    assert silent["fingerprint"] is None  # no key, no read: the off switch

    asked = module.process(
        Job(
            module="identity",
            payload={**payload, "fingerprint": {"min_px": 32, "chroma_floor": 12.0}},
        )
    )
    assert asked["embeddings"] == []
    assert asked["fingerprint"]["color"] == "red"
    json.dumps(asked)


# --------------------------------------------------------------------------
# Ingest writes the columns (the module never does), flag-gated
# --------------------------------------------------------------------------

CROP_JPEG = b"\xff\xd8car-crop-bytes"


class StubCarDetector:
    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "car",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 90.0, 130.0],
                    "track_id": 3,
                    "zones": [],
                    "crop_jpeg": CROP_JPEG,
                }
            ]
        }


class StubFingerprintIdentity:
    """Returns the shape `modules/identity.py` produces and records what
    it was asked, so the flag's plumbing is observable end to end."""

    def __init__(self):
        self.fingerprint_requests: list[dict | None] = []

    def process(self, job):
        req = job.payload.get("fingerprint")
        self.fingerprint_requests.append(req)
        read = None
        if req:
            read = read_color(
                solid((0, 0, 220)),
                min_px=req["min_px"],
                chroma_floor=req["chroma_floor"],
            ).as_payload()
        return {"embeddings": [], "fingerprint": read}


def fingerprint_service(sample_video, tmp_path, *, enabled: bool):
    identity = IdentityConfig(vector_db_path=str(tmp_path / "vectors"))
    identity.fingerprint.enabled = enabled
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
        identity=identity,
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/fp.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubCarDetector())
    stub = StubFingerprintIdentity()
    dispatcher.register("identity", stub)
    return IngestService(config, dispatcher=dispatcher), stub


def test_with_the_flag_on_every_detection_row_carries_its_read(
    sample_video, tmp_path
):
    service, stub = fingerprint_service(sample_video, tmp_path, enabled=True)
    service.run_camera(service.config.cameras[0])

    assert stub.fingerprint_requests, "identity jobs ran"
    assert all(
        req == {"min_px": 32, "chroma_floor": 12.0}
        for req in stub.fingerprint_requests
    )
    with service.Session() as session:
        rows = session.query(Detection).all()
    assert rows
    # Frames before the event crossed the significance gate never reach
    # the identity job, so their rows are honestly unmeasured (all-NULL)
    # rather than guessed — one job, one read, one measured row.
    measured = [r for r in rows if r.color_name is not None]
    assert len(measured) == len(stub.fingerprint_requests)
    for row in measured:
        assert row.color_name == "red"
        assert row.color_confidence == pytest.approx(1.0)
        assert row.color_chroma == pytest.approx(220.0)
        assert row.color_reason is None
        # The floors and measurements persist with the verdict — the
        # whole point of the discipline (CLD-128).
        assert row.color_min_px == 32
        assert row.color_chroma_floor == pytest.approx(12.0)
        assert row.color_crop_px == 100
        assert row.color_saturation is not None
    for row in rows:
        if row.color_name is None:
            assert row.color_reason is None and row.color_chroma is None
            assert row.color_min_px is None and row.color_chroma_floor is None


def test_with_the_flag_off_nothing_is_asked_and_nothing_is_written(
    sample_video, tmp_path
):
    service, stub = fingerprint_service(sample_video, tmp_path, enabled=False)
    service.run_camera(service.config.cameras[0])

    assert stub.fingerprint_requests, "identity jobs still ran"
    assert all(req is None for req in stub.fingerprint_requests)
    with service.Session() as session:
        rows = session.query(Detection).all()
    assert rows
    assert all(
        r.color_name is None and r.color_reason is None and r.color_chroma is None
        for r in rows
    )


# --------------------------------------------------------------------------
# The event page: chip when flagged on, nothing otherwise
# --------------------------------------------------------------------------


@pytest.fixture
def web(tmp_path):
    def build(*, enabled: bool):
        config = SiteConfig(
            site_id="test-site",
            cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
            storage=StorageConfig(
                db_url=f"sqlite:///{tmp_path}/web.db",
                media_dir=str(tmp_path / "media"),
            ),
        )
        config.identity.enabled = False
        config.identity.fingerprint.enabled = enabled
        engine = make_engine(config.storage.db_url)
        init_db(engine)
        Session = get_session(engine)
        with Session() as session:
            if session.get(Camera, "cam1") is None:
                session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
            if not session.query(Event).count():
                event = Event(
                    camera_id="cam1",
                    track_id=1,
                    class_name="car",
                    first_seen=TS,
                    last_seen=TS + timedelta(minutes=2),
                    detection_count=2,
                    best_confidence=0.9,
                    confidence_sum=1.8,
                )
                session.add(event)
                session.flush()
                for color in ("white", "white"):
                    session.add(
                        Detection(
                            event_id=event.id,
                            timestamp=TS,
                            class_name="car",
                            confidence=0.9,
                            bbox="[10, 10, 90, 130]",
                            zones="[]",
                            color_name=color,
                            color_confidence=0.8,
                            color_chroma=100.0,
                        )
                    )
            session.commit()
            event_id = session.query(Event).first().id
        return SimpleNamespace(
            client=TestClient(create_app(config)), event_id=event_id
        )

    return build


def test_the_chip_row_renders_only_behind_the_flag(web):
    on = web(enabled=True)
    body = on.client.get(f"/events/{on.event_id}").text
    assert "white" in body
    assert "2/2 frames" in body
    assert "no plate read attempted" in body

    off = web(enabled=False)
    body = off.client.get(f"/events/{off.event_id}").text
    assert "2/2 frames" not in body
    assert "no plate read attempted" not in body


def test_an_unmeasured_event_gets_no_chip_at_all(web, tmp_path):
    """Flag on, but nothing was ever measured — a Frigate-consumed
    event with no Detection rows, identity off, or pre-flag history.
    The chip must not render, because everything it would say ("no
    plate read attempted") is a claim about work that never ran."""
    env = web(enabled=True)
    engine = make_engine(f"sqlite:///{tmp_path}/web.db")
    with get_session(engine)() as session:
        for d in session.query(Detection).all():
            d.color_name = None
            d.color_confidence = None
            d.color_chroma = None
            d.color_reason = None
        session.commit()
    body = env.client.get(f"/events/{env.event_id}").text
    assert "no plate read attempted" not in body
    assert "color unknown" not in body


def test_an_ir_visit_says_unknown_ir_never_gray(web, tmp_path):
    env = web(enabled=True)
    # Rewrite the reads as what the always-IR camera produces: measured,
    # grayscale, no color named.
    engine = make_engine(f"sqlite:///{tmp_path}/web.db")
    with get_session(engine)() as session:
        for d in session.query(Detection).all():
            d.color_name = None
            d.color_confidence = None
            d.color_chroma = 0.0
            d.color_reason = REASON_NO_CHROMA
        session.commit()
    body = env.client.get(f"/events/{env.event_id}").text
    assert "color unknown (IR)" in body
    assert "grayscale frames" in body
