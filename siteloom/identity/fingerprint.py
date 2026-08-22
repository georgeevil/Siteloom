"""Vehicle fingerprint attributes (CLD-254): crop -> color read.

Flock-style vehicle search needs human-readable attributes next to the
plate and the appearance vector. This module computes the first one —
body color — as pure pixel math on the crop the pipeline already has
(no model, no new dependency). Body type is free from the YOLO class
and plate status from existing `PlateRead` rows, so color is the only
new measurement.

The shape copies the plate-read discipline (CLD-85/CLD-128): measure
first, gate after, and record the measurements *and* the floors they
were judged against on the row — so moving a floor later is a question
about existing data, never a reason to re-run anything. A read that
produces no color still travels, carrying its reason.

Two honesty rules the numbers must keep:

* **An achromatic crop names no color.** An IR frame (the front-yard
  camera never leaves IR) is pure grayscale, and naming the gray "gray"
  would be a confidently wrong answer that poisons any later attribute
  search. Chroma is measured over the whole crop — `crop_margin` grows
  the crop past the bbox, so a daylight crop carries chromatic
  background even around a white car, while an IR crop has none
  anywhere. Below the floor the read is `no-chroma`: *cannot measure
  color here*, deliberately covering both IR and the rare all-white
  daylight crop rather than guessing between them.
* **A tiny crop names no color.** Under `min_px` the center region is a
  handful of pixels and the vote is noise; `too-small` mirrors the
  plate-width floor.

Everything returned is scalars — the payload crosses a process boundary
under a Celery/Ray backend, same contract as `PlateRead.as_payload`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

#: Rejection reasons recorded on a read that names no color.
REASON_TOO_SMALL = "too-small"  # crop under the pixel floor
REASON_NO_CHROMA = "no-chroma"  # grayscale crop (IR, or nothing chromatic)

#: Hue bin edges on OpenCV's 0..179 hue circle. Red wraps, so it is the
#: leftover outside these bins rather than a bin of its own.
_HUE_BINS: tuple[tuple[str, int, int], ...] = (
    ("orange", 10, 22),
    ("yellow", 22, 34),
    ("green", 34, 78),
    ("blue", 78, 128),
    ("purple", 128, 155),
)

#: Center fraction of the crop the color vote runs over. The margin the
#: crop carries past the bbox is background by construction; the vehicle
#: body is the middle.
_CENTER_FRACTION = 0.6

#: Per-pixel saturation below this (0..255) votes an achromatic name
#: (white/black/gray by value) instead of a hue bin.
_ACHROMATIC_SATURATION = 60


@dataclass
class ColorRead:
    """One color measurement on one crop.

    `color` is None when nothing was named; `reason` says why. The
    measurements (`chroma_p95`, `saturation`) and the floors applied
    (`min_px`, `chroma_floor`) ride along whether or not a color was
    named, so the floors are chosen by reading the table (CLD-128's
    rule for plates, kept here).
    """

    color: str | None = None
    #: Fraction of center-region pixels that voted the winning name.
    confidence: float | None = None
    #: 95th percentile of per-pixel channel spread over the whole crop —
    #: the grayscale measure the no-chroma floor is judged on.
    chroma_p95: float | None = None
    #: Mean saturation (0..1) over the center region.
    saturation: float | None = None
    reason: str | None = None
    min_px: int | None = None
    chroma_floor: float | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "confidence": self.confidence,
            "chroma_p95": self.chroma_p95,
            "saturation": self.saturation,
            "reason": self.reason,
            "min_px": self.min_px,
            "chroma_floor": self.chroma_floor,
        }


def _pixel_color_names(center_bgr: np.ndarray) -> np.ndarray:
    """Name every center pixel: hue bin when saturated, value tier when
    not. Vectorized — a per-pixel Python loop on a 200px crop is the
    kind of cost that does not belong in the ingest path."""
    hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)

    names = np.full(h.shape, "red", dtype=object)
    for name, lo, hi in _HUE_BINS:
        names[(h >= lo) & (h < hi)] = name
    # Brown is dark orange — a separate perceptual color that shares a
    # hue band, which is why it is a value split rather than a bin.
    names[(names == "orange") & (v < 130)] = "brown"

    achromatic = s < _ACHROMATIC_SATURATION
    names[achromatic & (v >= 170)] = "white"
    names[achromatic & (v < 80)] = "black"
    names[achromatic & (v >= 80) & (v < 170)] = "gray"
    return names


def read_color(
    crop_bgr: np.ndarray, *, min_px: int, chroma_floor: float
) -> ColorRead:
    """Measure the crop and name its dominant color, or say why not."""
    height, width = crop_bgr.shape[:2]
    if min(height, width) < min_px:
        return ColorRead(reason=REASON_TOO_SMALL, min_px=min_px)

    # Chroma over the WHOLE crop, margin included: the background is
    # what separates "white car in daylight" from "IR frame".
    as_int = crop_bgr.astype(np.int32)
    spread = as_int.max(axis=2) - as_int.min(axis=2)
    chroma_p95 = float(np.percentile(spread, 95))

    cy, cx = int(height * (1 - _CENTER_FRACTION) / 2), int(
        width * (1 - _CENTER_FRACTION) / 2
    )
    center = crop_bgr[cy : height - cy or None, cx : width - cx or None]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[..., 1].mean() / 255.0)

    if chroma_p95 < chroma_floor:
        return ColorRead(
            chroma_p95=chroma_p95,
            saturation=saturation,
            reason=REASON_NO_CHROMA,
            min_px=min_px,
            chroma_floor=chroma_floor,
        )

    names = _pixel_color_names(center)
    values, counts = np.unique(names, return_counts=True)
    winner = int(counts.argmax())
    return ColorRead(
        color=str(values[winner]),
        confidence=float(counts[winner] / names.size),
        chroma_p95=chroma_p95,
        saturation=saturation,
        min_px=min_px,
        chroma_floor=chroma_floor,
    )


@dataclass
class VisitColor:
    """Display-time consensus over one visit's per-frame reads.

    Grouping is display-only, the same decision `/plates` made
    (CLD-131): the per-frame rows stay — they are what the floors are
    tuned from — and this is only how a screen summarizes them.
    """

    color: str | None
    #: Frames that voted the winning color / frames that named any color.
    agreeing: int
    named: int
    #: Frames whose read gave no color, by reason (e.g. all-IR visits
    #: show up here as {"no-chroma": n} — the honest "unknown (IR)").
    unnamed_reasons: dict[str, int]


def visit_color(
    reads: list[tuple[str | None, float | None, str | None]],
) -> VisitColor | None:
    """Majority color over (color, confidence, reason) per-frame rows.

    Confidence-weighted so ten half-hearted "gray" frames do not outvote
    six unanimous "white" ones. Returns None when nothing was measured
    at all (fingerprinting off, or pre-column rows) — a screen renders
    that as nothing, never as "unknown".
    """
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    unnamed: dict[str, int] = {}
    named = 0
    for color, confidence, reason in reads:
        if color is not None:
            named += 1
            weights[color] = weights.get(color, 0.0) + (confidence or 0.0)
            counts[color] = counts.get(color, 0) + 1
        elif reason is not None:
            unnamed[reason] = unnamed.get(reason, 0) + 1
    if not named and not unnamed:
        return None
    if not named:
        return VisitColor(color=None, agreeing=0, named=0, unnamed_reasons=unnamed)
    winner = max(weights, key=lambda k: weights[k])
    return VisitColor(
        color=winner,
        agreeing=counts[winner],
        named=named,
        unnamed_reasons=unnamed,
    )
