"""Plate detection + OCR (PRD §6.4, plate path).

Optional dependency group `plates` (onnx-based, no GPU needed):
    pip install -r requirements-plates.txt
If the packages are missing the vehicle path silently degrades to
visual re-ID only — plates are an enhancement, never a requirement.

`read()` returns a `PlateRead` describing the **whole attempt**, not just
its verdict (CLD-85). It used to return `str | None`, which threw away
everything anyone would need to judge the OCR: the detector's confidence
picked a box and was discarded, no OCR confidence was captured,
`normalize_plate` is lossy and irreversible, and a short read returned
None with no record that a read had happened at all. Motorcycle plates
are exactly the short/angled/rear-only case that falls under that bar,
so "how is plate OCR doing on motorcycles?" (CLD-9) was unanswerable
from the database.

Two invariants shape the return type:

* **This module is compute.** It may not write to SQLite or the vector
  store — the resolver owns identity state and ingest owns the rows. So
  a read is *returned*, and ingest persists it as a `PlateRead` row next
  to the Detection and EventIdentity rows it already writes.
* **A module result must stay serializable.** The result crosses a
  process boundary under a future Celery/Ray backend, so the plate
  sub-crop travels as JPEG **bytes**, never as an ndarray, and
  `as_payload()` flattens the dataclass to a plain dict for the trip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

_PLATE_RE = re.compile(r"[^A-Z0-9]")

#: Shortest normalized read accepted as a plate, unless configured
#: otherwise (`IdentifierConfig.plate_min_chars`). A floor, not a truth:
#: it is the exact bar short motorcycle plates fall under, which is why
#: it is configuration and why a read that fails it is still recorded.
DEFAULT_MIN_CHARS = 4

#: Why a read produced no usable plate. A negative is data — the same
#: philosophy as `Annotation.rejected` keeping rejections rather than
#: deleting them — so every one of these is written down.
REASON_NO_BOX = "no-box"  # the plate detector found no region at all
REASON_EMPTY_CROP = "empty-crop"  # box fell outside the vehicle crop
REASON_NO_TEXT = "no-text"  # OCR returned nothing readable
REASON_TOO_SHORT = "too-short"  # normalized read is under the floor
REASONS = (REASON_NO_BOX, REASON_EMPTY_CROP, REASON_NO_TEXT, REASON_TOO_SHORT)


def normalize_plate(text: str) -> str:
    """Uppercase, strip everything that is not [A-Z0-9].

    Lossy and irreversible on purpose — this is the form plates are
    matched on. `PlateRead.raw_text` keeps what the OCR actually said, so
    a near-miss stays distinguishable from a clean read afterwards.
    """
    return _PLATE_RE.sub("", text.upper())


@dataclass(frozen=True)
class PlateRead:
    """One OCR attempt on one vehicle crop.

    Every field is a scalar or bytes: this is the shape that survives the
    trip to the application layer under any dispatch backend.
    """

    #: The accepted plate (normalized), or None when the attempt failed.
    #: This is the only field the resolver ever sees — plate-first
    #: matching is unchanged by the instrumentation around it.
    text: str | None
    #: Exactly what the OCR returned, before normalization.
    raw_text: str | None = None
    #: `normalize_plate(raw_text)`, kept even when it fell under the
    #: floor: lowering `min_chars` later must not require re-running OCR.
    normalized: str | None = None
    #: Confidence of the plate-region box this read came from.
    detector_confidence: float | None = None
    #: Mean per-character OCR probability, when the installed
    #: fast-plate-ocr exposes one. None means "not reported" — never a
    #: stand-in number.
    ocr_confidence: float | None = None
    #: The plate sub-region as JPEG bytes — a *third* image, distinct
    #: from the detection crop, so a human can see what the OCR saw.
    #: Never an ndarray (dispatch invariant).
    plate_jpeg: bytes | None = None
    #: One of REASONS when the attempt produced no plate, else None.
    reason: str | None = None
    #: The floor this read was judged against, recorded so a row stays
    #: readable without knowing what the config said that night.
    min_chars: int = DEFAULT_MIN_CHARS

    @property
    def accepted(self) -> bool:
        return self.text is not None

    def as_payload(self) -> dict[str, Any]:
        """Plain-dict form, for embedding in a module result."""
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "normalized": self.normalized,
            "detector_confidence": self.detector_confidence,
            "ocr_confidence": self.ocr_confidence,
            "plate_jpeg": self.plate_jpeg,
            "reason": self.reason,
            "min_chars": self.min_chars,
        }


def mean_confidence(probs: Any) -> float | None:
    """Per-character probabilities reduced to one number, or None.

    None when the library reported nothing: an absent confidence must
    read as absent, not as zero — the "a rate never travels without its
    denominator" rule from stats.py, one layer down.
    """
    if probs is None:
        return None
    try:
        arr = np.asarray(probs, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return float(arr.mean())


def parse_ocr_result(result: Any) -> tuple[str | None, float | None]:
    """(raw text, confidence) out of whatever fast-plate-ocr returned.

    The pinned version's shape is not the only one the dependency range
    allows, and the two disagree: 1.0's `run()` returns `list[str]` (or
    `(list[str], char_probs)` with `return_confidence=True`), while 1.1
    returns `list[PlatePrediction]` carrying `plate` and `char_probs`.
    Both are read here rather than inline in `read()`, because getting it
    wrong is silent — the old code's `str(texts)` fallback would happily
    have normalized a repr into a plate.
    """
    probs: Any = None
    if isinstance(result, tuple):  # 1.0 with return_confidence=True
        texts = result[0] if len(result) > 0 else None
        probs = result[1] if len(result) > 1 else None
    else:
        texts = result
    if isinstance(texts, (list, tuple)):
        first = texts[0] if len(texts) > 0 else None
    else:
        first = texts
    if probs is not None and not isinstance(probs, (str, bytes)):
        try:
            probs = probs[0] if len(probs) > 0 else None
        except TypeError:
            probs = None
    if first is None:
        return None, None
    if isinstance(first, str):
        return (first or None), mean_confidence(probs)
    # 1.1's PlatePrediction, or anything else shaped like it.
    text = getattr(first, "plate", None)
    if not isinstance(text, str):
        # An unrecognized shape reads as "no text" rather than as a
        # stringified object: a repr that normalizes to four characters
        # would enter the identity store as a plate.
        log.warning("unrecognized OCR result %r; treating as no text", type(first))
        return None, None
    return (text or None), mean_confidence(getattr(first, "char_probs", probs))


def encode_plate_crop(plate_bgr: np.ndarray) -> bytes | None:
    """The plate sub-region as JPEG bytes, or None if it will not encode.

    Deliberately its own image. `crop_jpeg` is doing two jobs already —
    the display thumbnail *and* the embedder input — and changing it
    invalidates every stored vector, so the evidence image for an OCR
    read is a third file that leaves it untouched.
    """
    ok, buf = cv2.imencode(".jpg", plate_bgr)
    if not ok:
        return None
    return bytes(buf.tobytes())


class PlateReader:
    """Detect the plate region in a vehicle crop, then OCR it."""

    def __init__(self) -> None:
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import LicensePlateDetector

        self._detector = LicensePlateDetector(
            detection_model="yolo-v9-t-384-license-plate-end2end"
        )
        self._ocr = LicensePlateRecognizer("cct-xs-v1-global-model")

    def _run_ocr(self, plate_bgr: np.ndarray) -> tuple[str | None, float | None]:
        """OCR one plate crop, asking for confidence where it is offered."""
        try:
            result = self._ocr.run(plate_bgr, return_confidence=True)
        except TypeError:  # a build whose run() takes no such keyword
            result = self._ocr.run(plate_bgr)
        return parse_ocr_result(result)

    def read(
        self, vehicle_bgr: np.ndarray, *, min_chars: int = DEFAULT_MIN_CHARS
    ) -> PlateRead:
        """One attempt, always described — success or failure.

        No extra inference is bought here: the detector pass and the OCR
        pass are the ones that already ran on this crop. Everything added
        is capture.
        """
        detections = self._detector.predict(vehicle_bgr)
        if not detections:
            return PlateRead(text=None, reason=REASON_NO_BOX, min_chars=min_chars)
        best = max(detections, key=lambda d: d.confidence)
        confidence = float(best.confidence)
        bb = best.bounding_box
        plate = vehicle_bgr[
            max(0, int(bb.y1)) : int(bb.y2), max(0, int(bb.x1)) : int(bb.x2)
        ]
        if plate.size == 0:
            return PlateRead(
                text=None,
                detector_confidence=confidence,
                reason=REASON_EMPTY_CROP,
                min_chars=min_chars,
            )
        jpeg = encode_plate_crop(plate)
        raw, ocr_confidence = self._run_ocr(plate)
        if not raw:
            return PlateRead(
                text=None,
                detector_confidence=confidence,
                ocr_confidence=ocr_confidence,
                plate_jpeg=jpeg,
                reason=REASON_NO_TEXT,
                min_chars=min_chars,
            )
        normalized = normalize_plate(raw)
        # Junk reads are rejected, not hidden: the row keeps the raw text,
        # the normalized text and the floor it was judged against, so
        # moving the floor and re-reading the table answers CLD-9 without
        # re-running any inference.
        too_short = len(normalized) < min_chars
        return PlateRead(
            text=None if too_short else normalized,
            raw_text=raw,
            normalized=normalized,
            detector_confidence=confidence,
            ocr_confidence=ocr_confidence,
            plate_jpeg=jpeg,
            reason=REASON_TOO_SHORT if too_short else None,
            min_chars=min_chars,
        )


def try_build_plate_reader() -> PlateReader | None:
    try:
        return PlateReader()
    except ImportError:
        log.warning(
            "plate OCR requested but dependencies missing — "
            "install with `pip install -r requirements-plates.txt`; continuing with visual re-ID only"
        )
        return None
    except Exception as exc:  # model download failures etc.
        log.warning("plate OCR unavailable (%s); continuing without it", exc)
        return None
