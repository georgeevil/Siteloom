#!/usr/bin/env python
"""Build a synthetic media corpus for the resumability runbook (CLI-11).

The interrupt/resume paths can only be exercised over a corpus big
enough to stop in the middle of, and the repo ships two sample files.
This derives one from `samples/bus.jpg`: single-face crops and the
two-face original, jittered so every file is a distinct image, plus
Google-Takeout-shaped JSON sidecars carrying `people` tags.

    python scripts/make_resume_corpus.py OUT --count 400

Produces OUT/library/ (flat images+videos, for `siteloom library`) and
OUT/takeout/Google Photos/Photos from 2024/ (images + sidecars, for
`siteloom takeout import`).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NAMES = ["Ada Lovelace", "Grace Hopper"]


def face_crops(image: np.ndarray) -> list[np.ndarray]:
    """One crop per face in the source photo, generously padded."""
    from siteloom.identity.embedders import FaceEmbedder

    crops = []
    h, w = image.shape[:2]
    for face in FaceEmbedder().detect(image):
        x, y, fw, fh = (float(v) for v in face[:4])
        pad = max(fw, fh) * 1.2
        x1, y1 = int(max(0, x - pad)), int(max(0, y - pad))
        x2, y2 = int(min(w, x + fw + pad)), int(min(h, y + fh + pad))
        crop = image[y1:y2, x1:x2]
        if crop.size:
            crops.append(crop)
    return crops


def jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make each copy a distinct file without breaking face detection."""
    out = cv2.convertScaleAbs(
        image, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-12, 12)
    )
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(
        -3, 4, out.shape, dtype=np.int16
    )
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def sidecar(path: Path, people: list[str], taken: int) -> None:
    path.with_suffix(path.suffix + ".supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": path.name,
                "photoTakenTime": {"timestamp": str(taken)},
                "people": [{"name": n} for n in people],
            }
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=400, help="images per corpus")
    ap.add_argument("--videos", type=int, default=0, help="video copies (library only)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    source = cv2.imread(str(REPO / "samples" / "bus.jpg"))
    if source is None:
        raise SystemExit("samples/bus.jpg not found — run from the repo root")
    crops = face_crops(source)
    if len(crops) < 2:
        raise SystemExit(f"expected 2 faces in bus.jpg, found {len(crops)}")

    rng = random.Random(args.seed)
    library = args.out / "library"
    takeout = args.out / "takeout" / "Google Photos" / "Photos from 2024"
    for d in (library, takeout):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        # Two thirds single-face (1 face + 1 name = pass-1 "certain",
        # which is what seeds the gallery); one third the two-face
        # original, which only pass 2 can name.
        if i % 3 < 2:
            who = i % 2
            image, people = crops[who], [NAMES[who]]
        else:
            image, people = source, list(NAMES)
        image = jitter(image, rng)
        name = f"IMG_{i:05d}.jpg"
        cv2.imwrite(str(library / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(takeout / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sidecar(takeout / name, people, 1_700_000_000 + i * 3600)

    for i in range(args.videos):
        shutil.copy(REPO / "samples" / "sample.mp4", library / f"CLIP_{i:04d}.mp4")

    print(f"library: {len(list(library.iterdir()))} files -> {library}")
    print(f"takeout: {args.count} images + sidecars -> {takeout}")


if __name__ == "__main__":
    main()
