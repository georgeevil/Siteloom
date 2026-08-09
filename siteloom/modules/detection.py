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

`bbox` is the detection box; `crop_jpeg` is that box grown by
DetectionConfig.crop_margin so the crop carries some surrounding frame.
One crop serves both jobs — it is stored for display and it is what the
identity embedders see — so the two can never disagree about what a
detection looked like.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from siteloom.config import DetectionConfig
from siteloom.dispatch.base import Job

#: ultralytics' bytetrack.yaml defaults, restated here so the effective
#: tracker config is explicit and stable across library upgrades.
#: DetectionConfig.tracker entries are merged over these.
#:
#: `track_buffer` deliberately departs from ultralytics' 30 (CLD-96).
#: It counts *sampled* frames, and their default assumes 30 fps video —
#: at the few-fps Siteloom samples, 30 frames is six seconds of a lost
#: track coasting on a Kalman prediction whose covariance has grown
#: enormous, ready to adopt whoever next appears near it. That is not
#: hypothetical: it merged two different people into one 108-detection
#: event on real footage, bridging a 6.2 s hole with a 165 px jump.
#: 10 frames is ~2 s at 5 fps, which measurement put at a 3.0 s / 16 px
#: worst case over the same clip — a distance well inside one box width,
#: i.e. plausibly still the same person.
#:
#: Raising it back is reasonable for a camera sampled at a higher rate;
#: it is config, not a constant. What must not happen is treating the
#: upstream 30 as neutral when the sampling rate makes it six seconds.
TRACKER_DEFAULTS: dict[str, Any] = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 10,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def tracker_config_path(cfg: DetectionConfig) -> Path:
    """Materialize the effective tracker config as a YAML file.

    ultralytics only takes tracker settings as a file path, so the merged
    dict is written under the model cache, named by content hash — the
    same config always maps to the same file, and a config change never
    reuses a stale one.
    """
    merged = {**TRACKER_DEFAULTS, **cfg.tracker}
    text = yaml.safe_dump(merged, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = (
        Path.home() / ".cache" / "siteloom" / "trackers" / f"tracker-{digest}.yaml"
    )
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


class DetectionModule:
    def __init__(self, cfg: DetectionConfig | None = None):
        self.cfg = cfg or DetectionConfig()
        # One YOLO instance per camera: ultralytics keeps ByteTrack state
        # on the predictor, so sharing one instance across cameras would
        # tangle their track IDs. The nano model is small enough that a
        # per-camera copy is cheap, and each stays warm-loaded (the Ray
        # actor pattern, PRD §7).
        self._models: dict[str, Any] = {}
        self._tracker_path = tracker_config_path(self.cfg)

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
        # Run the detector at the lowest applicable threshold; per-class
        # minimums are applied to its output below. (YOLO's conf is a
        # single global floor.)
        floor = min(
            [self.cfg.confidence, *self.cfg.class_confidence.values()]
        )
        results = model.track(
            image,
            persist=True,
            conf=floor,
            device=self.cfg.device,
            tracker=str(self._tracker_path),
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
            confidence = float(boxes.conf[i])
            if confidence < self.cfg.class_confidence.get(
                class_name, self.cfg.confidence
            ):
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            track_id = int(boxes.id[i]) if boxes.id is not None else None

            hit_zones = _zones_hit(zones, (x1, y1, x2, y2), w, h)
            if require_zone and not hit_zones:
                continue

            cx1, cy1, cx2, cy2 = expand_box(
                (x1, y1, x2, y2), self.cfg.crop_margin, w, h
            )
            crop = image[cy1:cy2, cx1:cx2]
            crop_jpeg = None
            if crop.size:
                ok, buf = cv2.imencode(".jpg", crop)
                crop_jpeg = buf.tobytes() if ok else None
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                    "zones": hit_zones,
                    "crop_jpeg": crop_jpeg,
                }
            )
        return {"detections": detections}


def expand_box(
    bbox: tuple[float, float, float, float],
    margin: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Grow a bbox by `margin` of its own size on each side, clamped to the frame.

    The margin is a fraction of each dimension, so a tall person crop and
    a wide vehicle crop keep their shape and just gain context. Bounds are
    floored/ceiled outward — rounding must never eat into the box itself —
    and the result is always a valid slice (x1 <= x2, y1 <= y2), which is
    what makes the degenerate box at a frame edge an empty crop rather
    than an exception inside cv2.
    """
    x1, y1, x2, y2 = bbox
    dx = max(0.0, x2 - x1) * margin
    dy = max(0.0, y2 - y1) * margin
    nx1 = int(math.floor(max(0.0, x1 - dx)))
    ny1 = int(math.floor(max(0.0, y1 - dy)))
    nx2 = int(math.ceil(min(float(width), x2 + dx)))
    ny2 = int(math.ceil(min(float(height), y2 + dy)))
    return nx1, ny1, max(nx1, nx2), max(ny1, ny2)


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
