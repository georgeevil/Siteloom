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
    {"embeddings": [{identifier, algo, vector: [float], plate: str|None}]}
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
            if ident.plate_ocr:
                reader = self._get_plate_reader()
                if reader is not None:
                    plate = reader.read(crop)
            if vector is None and plate is None:
                continue  # e.g. no face found in the person crop
            out.append(
                {
                    "identifier": key,
                    "algo": ident.algo,
                    "vector": vector.tolist() if vector is not None else None,
                    "plate": plate,
                }
            )
        return {"embeddings": out}
