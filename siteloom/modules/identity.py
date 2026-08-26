"""Identity module: crop -> embeddings (+ plate text).

Compute only — this module never touches the vector store or SQLite.
It returns serializable embeddings; the IdentityResolver (application
layer, central by design) does matching and identity creation. That
split is what NFR2 requires: on a distributed fleet the edge workers
run THIS code and only embeddings/metadata travel upstream.

Job payload:
    crop_jpeg:  bytes — the detection crop (first-pass filter output)
    class_name: str
    plate_floors: {identifier: {min_chars, min_width_px, min_sharpness,
                   min_char_confidence}} — optional; the floors resolved
                   by `IdentityConfig.plate_floors_for` for the camera
                   this crop came from (CLD-128). Absent keys fall back
                   to the identifier's site-wide values, which keeps a
                   directly-driven module (tests, replay) working.
    skip_plate_ocr: [identifier, ...] — optional; identifiers whose OCR
                   ingest rationed out for this frame (CLD-130's cadence
                   cap). The embedding still runs — only the OCR is
                   skipped, and no PlateRead travels for it.
    fingerprint: {min_px, chroma_floor} — optional; when present, the
                   vehicle-fingerprint color read (CLD-254) runs on the
                   crop with these floors. Ingest sends it only when the
                   feature flag is on and the class is fingerprinted, so
                   its absence is the off switch.
    appearance_only: bool — optional; when true, skip identifiers, OCR
                   and fingerprinting entirely and return one generic
                   appearance embedding of the whole crop under the
                   reserved identifier "_appearance". For callers that
                   compare crops to each other (the occlusion swap
                   check) rather than resolving them.
Result:
    {"embeddings": [{identifier, algo, vector: [float],
                     quality: float|None, plate: str|None,
                     plate_read: dict|None}],
     "fingerprint": dict|None}

`quality` is the embedder's own confidence in what it embedded, where it
has one — the face pipeline reports YuNet's score for the face it chose.
None for embedders with no such signal; ingest then falls back to the
detector's box confidence, the pre-existing behaviour.

`fingerprint` is per crop, not per identifier — color belongs to the
detection, and it travels (as `ColorRead.as_payload()`) even when the
read named no color, exactly as failed plate reads do.

`plate_read` is the whole OCR attempt (CLD-85), flattened to a plain dict
by `PlateRead.as_payload()` — scalars plus the plate sub-crop as JPEG
**bytes**. Nothing here may be an ndarray or a live handle: this result
crosses a process boundary under a Celery/Ray backend. Ingest writes the
row; this module still writes nothing anywhere.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from siteloom.config import IdentityConfig
from siteloom.dispatch.base import Job
from siteloom.identity.registry import IdentifierRegistry


class IdentityModule:
    def __init__(self, cfg: IdentityConfig, device: str = "mps"):
        self.registry = IdentifierRegistry(cfg, device=device)
        self._plate_reader = None
        self._plate_reader_tried = False

    def _get_plate_reader(self):
        if not self._plate_reader_tried:
            self._plate_reader_tried = True
            from siteloom.identity.plates import try_build_plate_reader

            self._plate_reader = try_build_plate_reader()
        return self._plate_reader

    def process(self, job: Job) -> dict[str, Any]:
        payload = job.payload
        crop = cv2.imdecode(
            np.frombuffer(payload["crop_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if crop is None:
            raise ValueError("could not decode crop_jpeg")

        if payload.get("appearance_only"):
            # A bare appearance embedding of the whole crop, regardless
            # of which identifiers apply to the class — the occlusion
            # swap check compares crops *to each other*, so it needs a
            # like-for-like vector even where the configured identifier
            # is face-only (a distant person's face never resolves) and
            # must not pay for OCR or face detection it will not use.
            # "_appearance" is not a registry key on purpose: nothing
            # downstream may mistake this for a resolvable identifier.
            vector = self.registry.generic_embedder().embed(crop)
            return {
                "embeddings": [{
                    "identifier": "_appearance",
                    "algo": "generic",
                    "vector": vector.tolist() if vector is not None else None,
                    "quality": None,
                    "plate": None,
                    "plate_read": None,
                }],
                "fingerprint": None,
            }

        fingerprint = None
        fp_floors = payload.get("fingerprint")
        if fp_floors:
            from siteloom.identity.fingerprint import read_color

            fingerprint = read_color(
                crop,
                min_px=fp_floors["min_px"],
                chroma_floor=fp_floors["chroma_floor"],
            ).as_payload()

        out: list[dict[str, Any]] = []
        for key, ident in self.registry.identifiers_for(payload["class_name"]):
            embedder = self.registry.embedder_for(key)
            # Per-identifier quality, where the embedder can measure one:
            # the face pipeline knows how confident it is in the *face*
            # (YuNet's score), which is a different fact from the YOLO
            # confidence in the parent box that ingest otherwise uses.
            # A plain float — this dict crosses a process boundary.
            quality = None
            if hasattr(embedder, "embed_best"):
                vector, quality = embedder.embed_best(crop)
            else:
                vector = embedder.embed(crop)
            plate = None
            plate_read = None
            if ident.plate_ocr and key not in payload.get("skip_plate_ocr", ()):
                reader = self._get_plate_reader()
                if reader is not None:
                    # The floors the application layer resolved for this
                    # camera (CLD-128), else the identifier's site-wide
                    # values — same numbers, no camera named.
                    floors = (payload.get("plate_floors") or {}).get(key) or {}
                    read = reader.read(
                        crop,
                        min_chars=floors.get("min_chars", ident.plate_min_chars),
                        min_width=floors.get(
                            "min_width_px", ident.plate_min_width_px
                        ),
                        min_sharpness=floors.get(
                            "min_sharpness", ident.plate_min_sharpness
                        ),
                        min_char_confidence=floors.get(
                            "min_char_confidence", ident.plate_min_char_confidence
                        ),
                    )
                    plate = read.text
                    plate_read = read.as_payload()
                    if not ident.plate_save_crops:
                        # Recording the attempt is cheap; keeping every
                        # plate crop is a disk-space decision an operator
                        # may reasonably decline. The row still gets its
                        # text, confidences and reason.
                        plate_read["plate_jpeg"] = None
            if vector is None and plate is None and plate_read is None:
                continue  # e.g. no face found in the person crop
            out.append(
                {
                    "identifier": key,
                    "algo": ident.algo,
                    "vector": vector.tolist() if vector is not None else None,
                    "quality": quality,
                    "plate": plate,
                    # A failed read still travels: the entry may carry
                    # nothing resolvable and exist only so ingest can
                    # record the attempt (see `_identify`).
                    "plate_read": plate_read,
                }
            )
        return {"embeddings": out, "fingerprint": fingerprint}
