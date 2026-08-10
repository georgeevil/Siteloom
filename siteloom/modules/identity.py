"""Identity module: crop -> embeddings (+ plate text).

Compute only — this module never touches the vector store or SQLite.
It returns serializable embeddings; the IdentityResolver (application
layer, central by design) does matching and identity creation. That
split is what NFR2 requires: on a distributed fleet the edge workers
run THIS code and only embeddings/metadata travel upstream.

Job payload:
    crop_jpeg:  bytes — the detection crop (first-pass filter output)
    class_name: str
Result:
    {"embeddings": [{identifier, algo, vector: [float], plate: str|None,
                     plate_read: dict|None}]}

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

        out: list[dict[str, Any]] = []
        for key, ident in self.registry.identifiers_for(payload["class_name"]):
            embedder = self.registry.embedder_for(key)
            vector = embedder.embed(crop)
            plate = None
            plate_read = None
            if ident.plate_ocr:
                reader = self._get_plate_reader()
                if reader is not None:
                    read = reader.read(crop, min_chars=ident.plate_min_chars)
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
                    "plate": plate,
                    # A failed read still travels: the entry may carry
                    # nothing resolvable and exist only so ingest can
                    # record the attempt (see `_identify`).
                    "plate_read": plate_read,
                }
            )
        return {"embeddings": out}
