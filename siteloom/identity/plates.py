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
REASON_TOO_SMALL = "too-small"  # plate region under the pixel-width floor
REASON_TOO_BLURRY = "too-blurry"  # plate region under the sharpness floor
REASON_TOO_SHORT = "too-short"  # normalized read is under the floor
REASON_LOW_CONFIDENCE = "low-confidence"  # weakest character under the floor
REASONS = (
    REASON_NO_BOX,
    REASON_EMPTY_CROP,
    REASON_NO_TEXT,
    REASON_TOO_SMALL,
    REASON_TOO_BLURRY,
    REASON_TOO_SHORT,
    REASON_LOW_CONFIDENCE,
)


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
    #: The *weakest* character's probability from the same array the mean
    #: is taken over. The mean is a poor rejection signal precisely
    #: because it hides a single substituted character: five characters
    #: at 0.98 and one at 0.35 average to 0.87, which is the "OCR was
    #: confident and still wrong" row. The minimum is the number that
    #: sees it, so it is what `min_char_confidence` gates on.
    ocr_min_confidence: float | None = None
    #: Size of the plate region **in the source pixels**, not of the
    #: vehicle crop. Plate width is the closest thing LPR has to a
    #: hardware spec — under roughly 100 px the characters carry too few
    #: pixels to be distinguished and the OCR is interpolating.
    plate_width: int | None = None
    plate_height: int | None = None
    #: Variance of the Laplacian over the plate region: the standard
    #: blur measure, and the one that separates "small but crisp" from
    #: "large and smeared by motion". Scale- and exposure-dependent, so
    #: it is a number to calibrate against your own cameras from the
    #: /plates table, never a universal constant.
    sharpness: float | None = None
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
            "ocr_min_confidence": self.ocr_min_confidence,
            "plate_width": self.plate_width,
            "plate_height": self.plate_height,
            "sharpness": self.sharpness,
            "plate_jpeg": self.plate_jpeg,
            "reason": self.reason,
            "min_chars": self.min_chars,
        }


def _prob_array(probs: Any) -> np.ndarray | None:
    """Per-character probabilities as a flat array, or None.

    None when the library reported nothing: an absent confidence must
    read as absent, not as zero — the "a rate never travels without its
    denominator" rule from stats.py, one layer down. Every reduction
    below inherits that, which is why none of them can turn "not
    reported" into a number that would trip a floor.
    """
    if probs is None:
        return None
    try:
        arr = np.asarray(probs, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def mean_confidence(probs: Any) -> float | None:
    """Per-character probabilities reduced to their mean, or None."""
    arr = _prob_array(probs)
    return None if arr is None else float(arr.mean())


def min_confidence(probs: Any) -> float | None:
    """The weakest character's probability, or None.

    Kept beside the mean rather than replacing it: the mean is the right
    summary of a read's overall quality and the minimum is the right
    rejection signal, and conflating them loses one of the two.
    """
    arr = _prob_array(probs)
    return None if arr is None else float(arr.min())


def laplacian_variance(image: np.ndarray) -> float | None:
    """How sharp an image is, by the variance of its Laplacian.

    High-frequency energy: a crisp plate has strong character edges, a
    motion-smeared one does not. Returns None for an image with no
    pixels rather than 0.0, which would read as "perfectly blurred".
    """
    if image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def gate_reason(
    *,
    normalized: str,
    plate_width: int | None,
    sharpness: float | None,
    char_confidence: float | None,
    min_chars: int,
    min_width: int,
    min_sharpness: float,
    min_char_confidence: float,
) -> str | None:
    """Which quality floor this read fails, or None if it clears them all.

    Pure, so the policy is testable without a model. Two rules are worth
    stating out loud:

    * **A metric that was never measured cannot fail a floor.** An OCR
      build that reports no per-character probabilities would otherwise
      have every read rejected the moment an operator set a confidence
      floor — an absent number read as zero, which is exactly the
      mistake `_prob_array` exists to prevent.
    * **Order is diagnosis, not arithmetic.** A 30-pixel plate that also
      read three characters is reported as `too-small`, because that is
      the fact an operator can act on (move the camera, zoom, or lower
      the floor knowingly); "too-short" would send them to the character
      floor instead, which is not what went wrong.
    """
    if min_width and plate_width is not None and plate_width < min_width:
        return REASON_TOO_SMALL
    if min_sharpness and sharpness is not None and sharpness < min_sharpness:
        return REASON_TOO_BLURRY
    if len(normalized) < min_chars:
        return REASON_TOO_SHORT
    if (
        min_char_confidence
        and char_confidence is not None
        and char_confidence < min_char_confidence
    ):
        return REASON_LOW_CONFIDENCE
    return None


def parse_ocr_result(result: Any) -> tuple[str | None, Any]:
    """(raw text, per-character probabilities) out of fast-plate-ocr.

    The probabilities come back unreduced: the caller wants both their
    mean (how the read looked overall) and their minimum (whether any
    one character was a guess), and a function that reduced here could
    only return one of them.

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
        return (first or None), probs
    # 1.1's PlatePrediction, or anything else shaped like it.
    text = getattr(first, "plate", None)
    if not isinstance(text, str):
        # An unrecognized shape reads as "no text" rather than as a
        # stringified object: a repr that normalizes to four characters
        # would enter the identity store as a plate.
        log.warning("unrecognized OCR result %r; treating as no text", type(first))
        return None, None
    return (text or None), getattr(first, "char_probs", probs)


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


#: The detector's logger, whose one shouted non-problem is quieted below.
_DETECTOR_LOG = "open_image_models.detection.core.yolo_v9.inference"


def quiet_empty_nms(record: logging.LogRecord) -> bool:
    """Drop the CoreML empty-NMS complaint, unless DEBUG is asked for.

    The plate detector's end-to-end graph does its NMS inside the model,
    and Apple's CoreML execution provider cannot run that subgraph when
    **no box survives**: it reports a zero-element tensor where it wants
    a dynamic shape (upstream onnxruntime#20372). ONNX Runtime logs it as
    an ERROR, open_image_models catches it and returns "no detections",
    and that is the correct answer — the crop had no plate in it.

    So a WARNING per plateless crop is a lie about severity, and there
    are a lot of them. Measured on 400 real crops from this site
    (2026-08-15): the error fired on 45, and on **none** of those did a
    CPU session — which never hits the bug — find a plate. Over the whole
    400 the two providers disagreed about exactly one crop, in the other
    direction, at the confidence threshold. Nothing is being lost.

    Matched on this failure's own words, because the same log line
    reports genuine inference failures too — a real one still arrives as
    a WARNING, loudly.

    Dropped rather than merely relabelled, and that distinction cost a
    round: demoting the record to DEBUG changes nothing on its own,
    because the logger's level was already cleared when `.warning()` was
    called and handlers here sit at NOTSET, so the line prints anyway
    with a different word in front of it.

    It survives when the *application* is running at DEBUG — asked of the
    root logger, not of the detector's own, which open_image_models pins
    at INFO. Keying on that one would put the escape hatch behind an
    incantation nobody would guess; keying on the root means
    `--log-level DEBUG`, which is what an operator reaches for, shows how
    often this really fires.
    """
    message = record.getMessage()
    if "CoreML" not in message or "zero elements" not in message:
        return True
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        return False
    record.levelno = logging.DEBUG
    record.levelname = "DEBUG"
    return True


def _build_detector() -> Any:
    """The plate detector, with its one known non-problem quieted.

    Two halves, because the complaint arrives twice by two routes: ONNX
    Runtime's own C++ logger writes to stderr (silenced per session, so
    other sessions — the OCR model's above all — keep every diagnostic
    they have), and open_image_models re-logs the caught exception
    through Python (demoted by the filter).

    CoreML is kept rather than sidestepped by pinning the session to CPU:
    measured on this site's crops, CPU costs 19.5 ms against CoreML's
    7.2 ms for the same answers, which is real throughput on every
    vehicle crop, spent to silence a log line.
    """
    import onnxruntime as ort
    from open_image_models.detection.factory import create_detector

    detector_log = logging.getLogger(_DETECTOR_LOG)
    if quiet_empty_nms not in detector_log.filters:
        detector_log.addFilter(quiet_empty_nms)

    options = ort.SessionOptions()
    options.log_severity_level = 4  # fatal only, this session alone
    return create_detector(
        "yolo-v9-t-384-license-plate-end2end", sess_options=options
    )


class PlateReader:
    """Detect the plate region in a vehicle crop, then OCR it."""

    def __init__(self) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        self._detector = _build_detector()
        self._ocr = LicensePlateRecognizer("cct-xs-v1-global-model")

    def _run_ocr(
        self, plate_bgr: np.ndarray
    ) -> tuple[str | None, float | None, float | None]:
        """OCR one plate crop: (text, mean confidence, weakest character)."""
        try:
            result = self._ocr.run(plate_bgr, return_confidence=True)
        except TypeError:  # a build whose run() takes no such keyword
            result = self._ocr.run(plate_bgr)
        raw, probs = parse_ocr_result(result)
        return raw, mean_confidence(probs), min_confidence(probs)

    def read(
        self,
        vehicle_bgr: np.ndarray,
        *,
        min_chars: int = DEFAULT_MIN_CHARS,
        min_width: int = 0,
        min_sharpness: float = 0.0,
        min_char_confidence: float = 0.0,
    ) -> PlateRead:
        """One attempt, always described — success or failure.

        No extra inference is bought here: the detector pass and the OCR
        pass are the ones that already ran on this crop. Everything added
        is capture — the quality metrics are two array reductions and one
        Laplacian over a crop measured in tens of pixels.

        The quality floors are applied **after** the OCR rather than
        before it, which costs one cheap inference on a plate already
        known to be too small. That is deliberate and it is the same
        trade `min_chars` makes: the row keeps the raw text, the
        normalized text, the measurements and the floors they were
        judged against, so "is 100 px the right bar?" is answered by
        moving it and re-reading the table, never by re-running a night
        of video that no longer exists.
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
        # Measured off the crop that was actually read, not off the box
        # coordinates: a box clipped at the crop's edge is smaller than
        # it claims, and the OCR saw the clipped version.
        height, width = plate.shape[:2]
        sharpness = laplacian_variance(plate)
        jpeg = encode_plate_crop(plate)
        raw, ocr_confidence, ocr_min = self._run_ocr(plate)
        if not raw:
            return PlateRead(
                text=None,
                detector_confidence=confidence,
                ocr_confidence=ocr_confidence,
                ocr_min_confidence=ocr_min,
                plate_width=width,
                plate_height=height,
                sharpness=sharpness,
                plate_jpeg=jpeg,
                reason=REASON_NO_TEXT,
                min_chars=min_chars,
            )
        normalized = normalize_plate(raw)
        # Junk reads are rejected, not hidden: the row keeps the raw text,
        # the normalized text and the floor it was judged against, so
        # moving the floor and re-reading the table answers CLD-9 without
        # re-running any inference.
        reason = gate_reason(
            normalized=normalized,
            plate_width=width,
            sharpness=sharpness,
            char_confidence=ocr_min,
            min_chars=min_chars,
            min_width=min_width,
            min_sharpness=min_sharpness,
            min_char_confidence=min_char_confidence,
        )
        return PlateRead(
            text=None if reason else normalized,
            raw_text=raw,
            normalized=normalized,
            detector_confidence=confidence,
            ocr_confidence=ocr_confidence,
            ocr_min_confidence=ocr_min,
            plate_width=width,
            plate_height=height,
            sharpness=sharpness,
            plate_jpeg=jpeg,
            reason=reason,
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
