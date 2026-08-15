"""CLD-129 spike: can channel spread separate IR from colour frames?

The claim to confirm before building anything: an IR frame is grayscale
by construction — the sensor forced all three channels to (nearly) the
same value — so a model-free statistic over a stored crop should split
IR from colour cleanly, with no inference cost and no false-positive
class. This measures two candidates over every detection crop on disk:

* **channel spread**: mean over pixels of (max(B,G,R) - min(B,G,R)),
  in 0..255. Exactly 0 on a pure-grayscale image; JPEG chroma noise
  keeps real IR crops slightly above 0.
* **saturation**: mean HSV S, 0..255. The same fact via the colour
  model; carried to check the two agree.

Usage:
    python measure_ir.py <media_dir> [out.csv]

Prints a per-camera, per-hour summary (hours are UTC — crop filenames
are store time, naive UTC by contract, CLD-100) and writes one CSV row
per crop for any further analysis. Plate sub-crops are skipped: the
question is about the camera's condition, and the vehicle crop is what
both the embedders and the plate reader actually receive.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def measure(path: Path) -> tuple[float, float] | None:
    image = cv2.imread(str(path))
    if image is None or image.ndim != 3:
        return None
    channels = image.astype(np.int16)
    spread = float((channels.max(axis=2) - channels.min(axis=2)).mean())
    saturation = float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].mean())
    return spread, saturation


def main() -> None:
    media = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ir_measurements.csv")
    rows = []
    for path in sorted(media.rglob("*.jpg")):
        if "plates" in path.parts:
            continue
        camera = path.relative_to(media).parts[0]
        stem = path.stem  # HHMMSS_micro_class_track
        hour = stem[0:2]
        measured = measure(path)
        if measured is None:
            continue
        spread, saturation = measured
        rows.append(
            {
                "camera": camera,
                "day": path.parent.name,
                "hour_utc": hour,
                "class": stem.split("_")[2] if len(stem.split("_")) > 2 else "",
                "file": str(path.relative_to(media)),
                "channel_spread": round(spread, 2),
                "saturation": round(saturation, 2),
            }
        )

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} crops -> {out}\n")

    by_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_bucket[(row["camera"], row["hour_utc"])].append(row["channel_spread"])
    print(f"{'camera':18} {'hour':>4} {'n':>5} {'p10':>7} {'median':>7} {'p90':>7}")
    for (camera, hour), values in sorted(by_bucket.items()):
        arr = np.array(values)
        print(
            f"{camera:18} {hour:>4} {len(values):>5}"
            f" {np.percentile(arr, 10):>7.1f}"
            f" {np.median(arr):>7.1f}"
            f" {np.percentile(arr, 90):>7.1f}"
        )

    spreads = np.array([r["channel_spread"] for r in rows])
    print("\noverall distribution of channel spread:")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:>2}: {np.percentile(spreads, q):7.2f}")
    # Where does the gap sit? Count crops per unit bin up to 30.
    hist, _ = np.histogram(spreads, bins=np.arange(0, 31))
    print("\ncrops per unit bin, spread 0..30:")
    print("  " + " ".join(f"{int(n)}" for n in hist))


if __name__ == "__main__":
    main()
