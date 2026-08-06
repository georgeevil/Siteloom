"""Object detection module (PRD §6.2).

First-pass filter: person/vehicle/bicycle/animal via YOLO with ByteTrack
track IDs. Downstream modules (face ID, plate/re-ID) will subscribe to
these detections by class instead of consuming raw frames — this module
is the reason expensive recognition never runs on empty frames.

Job payload (all serializable):
    image_jpeg: bytes        — the sampled frame, JPEG-encoded
    camera_id:  str
    timestamp:  str (ISO)
    zones:      [{name, points[[x,y]..]}]  — normalized polygons
    require_zone: bool

Result: {"detections": [{class_name, confidence, bbox, track_id,
                          zones, crop_jpeg}]}
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from siteloom.config import DetectionConfig
from siteloom.dispatch.base import Job


class DetectionModule:
    def __init__(self, cfg: DetectionConfig | None = None):
        self.cfg = cfg or DetectionConfig()
        # One YOLO instance per camera: ultralytics keeps ByteTrack state
        # on the predictor, so sharing one instance across cameras would
        # tangle their track IDs. The nano model is small enough that a
        # per-camera copy is cheap, and each stays warm-loaded (the Ray
        # actor pattern, PRD §7).
        self._models: dict[str, Any] = {}

    def _model_for(self, camera_id: str):
        model = self._models.get(camera_id)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(self.cfg.model)
            self._models[camera_id] = model
        return model

    def process(self, job: Job) -> dict[str, Any]:
        payload = job.payload
        image = cv2.imdecode(
            np.frombuffer(payload["image_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("could not decode image_jpeg")

        camera_id = payload["camera_id"]
        model = self._model_for(camera_id)
        results = model.track(
            image,
            persist=True,
            conf=self.cfg.confidence,
            device=self.cfg.device,
            verbose=False,
        )

        h, w = image.shape[:2]
        zones = payload.get("zones") or []
        require_zone = bool(payload.get("require_zone")) and bool(zones)
        wanted = set(self.cfg.classes)

        detections: list[dict[str, Any]] = []
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return {"detections": detections}

        for i in range(len(boxes)):
            class_name = names[int(boxes.cls[i])]
            if class_name not in wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            track_id = int(boxes.id[i]) if boxes.id is not None else None

            hit_zones = _zones_hit(zones, (x1, y1, x2, y2), w, h)
            if require_zone and not hit_zones:
                continue

            crop = image[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
            ok, crop_jpeg = cv2.imencode(".jpg", crop)
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": float(boxes.conf[i]),
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                    "zones": hit_zones,
                    "crop_jpeg": crop_jpeg.tobytes() if ok else None,
                }
            )
        return {"detections": detections}


def _zones_hit(
    zones: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[str]:
    """Return names of zones containing the bbox's bottom-center point.

    Bottom-center is where the object touches the ground — Frigate's
    convention — so a person whose head overlaps a zone but who is
    standing outside it doesn't trigger.
    """
    if not zones:
        return []
    x1, _y1, x2, y2 = bbox
    point = ((x1 + x2) / 2.0, y2)
    hits = []
    for zone in zones:
        poly = np.array(
            [(px * width, py * height) for px, py in zone["points"]], dtype=np.float32
        )
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            hits.append(zone["name"])
    return hits
