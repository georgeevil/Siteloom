"""Plate detection + OCR (PRD §6.4, plate path).

Optional dependency group `plates` (onnx-based, no GPU needed):
    pip install -r requirements-plates.txt
If the packages are missing the vehicle path silently degrades to
visual re-ID only — plates are an enhancement, never a requirement.
"""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

_PLATE_RE = re.compile(r"[^A-Z0-9]")


def normalize_plate(text: str) -> str:
    return _PLATE_RE.sub("", text.upper())


class PlateReader:
    """Detect the plate region in a vehicle crop, then OCR it."""

    def __init__(self) -> None:
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import LicensePlateDetector

        self._detector = LicensePlateDetector(
            detection_model="yolo-v9-t-384-license-plate-end2end"
        )
        self._ocr = LicensePlateRecognizer("cct-xs-v1-global-model")

    def read(self, vehicle_bgr: np.ndarray) -> str | None:
        detections = self._detector.predict(vehicle_bgr)
        if not detections:
            return None
        best = max(detections, key=lambda d: d.confidence)
        bb = best.bounding_box
        plate = vehicle_bgr[
            max(0, int(bb.y1)) : int(bb.y2), max(0, int(bb.x1)) : int(bb.x2)
        ]
        if plate.size == 0:
            return None
        texts = self._ocr.run(plate)
        if not texts:
            return None
        text = normalize_plate(texts[0] if isinstance(texts, list) else str(texts))
        # Reject junk reads — real plates have at least 4 characters.
        return text if len(text) >= 4 else None


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
