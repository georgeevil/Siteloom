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
    sample_fps: float — the camera's sampling rate, which sizes the
                tracker's lost-track buffer (track_buffer_s × fps);
                omitted by callers with no meaningful rate

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
#: Departures from ultralytics' bytetrack.yaml, each measured on the
#: tracker corpus (CLD-98; `docs/testing/tracker-corpus.md`):
#:
#: * **BoT-SORT with ReID over plain ByteTrack** (2026-08-25). ByteTrack
#:   matches on predicted position alone, which is exactly what fails
#:   when two subjects overlap: occlusion phantoms (mid/post-occlusion
#:   births) that the bridge metric cannot see because no track goes
#:   dark. `model: auto` reuses the detector's own backbone features, so
#:   there is no extra model or meaningful cost; `gmc_method: none`
#:   because the cameras are fixed and optical flow is pure cost.
#:
#: * **`new_track_thresh` 0.5, up from 0.25** (2026-08-25). The sliver
#:   of a half-hidden person is a low-confidence partial box; demanding
#:   more confidence to *found* a track than to continue one is what cut
#:   mid-occlusion births on the corpus, with no fragmentation cost.
#:
#: * **`track_buffer` derived from time, not fixed frames** (CLD-96).
#:   The knob counts *sampled* frames, so a fixed number changes meaning
#:   with the sampling rate — the same 10 is 2 s at 5 fps and 5 s at
#:   2 fps. The configured quantity is `DetectionConfig.track_buffer_s`
#:   (seconds); `tracker_config_path` derives the frame count when the
#:   caller supplies the camera's `sample_fps`. The 20 here is only the
#:   fallback for callers with no meaningful sampling rate (the library
#:   indexer's sparse frames, a bare DetectionModule in a test), equal
#:   to track_buffer_s=4.0 at the live cameras' 5 fps. An explicit
#:   `tracker.track_buffer` overrides both.
#:
#:   CLD-96 cut the buffer to ~2 s because a long buffer lets a dead
#:   track's coasting Kalman prediction adopt a stranger (two people
#:   merged into one 108-detection event, a 6.2 s hole bridged with a
#:   165 px jump). 4 s is safe *only because* re-acquisition is now
#:   verified by appearance — buffer length and ReID are one decision,
#:   not two. Do not raise this while turning `with_reid` off.
TRACKER_DEFAULTS: dict[str, Any] = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.5,
    "track_buffer": 20,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "none",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": True,
    "model": "auto",
}


def tracker_config_path(
    cfg: DetectionConfig, sample_fps: float | None = None
) -> Path:
    """Materialize the effective tracker config as a YAML file.

    ultralytics only takes tracker settings as a file path, so the merged
    dict is written under the model cache, named by content hash — the
    same config always maps to the same file, and a config change never
    reuses a stale one.

    `sample_fps` is what turns `track_buffer_s` (seconds, the configured
    quantity) into `track_buffer` (sampled frames, the tracker's unit);
    without it the TRACKER_DEFAULTS frame count stands. An explicit
    `tracker.track_buffer` wins over both.
    """
    merged = {**TRACKER_DEFAULTS, **cfg.tracker}
    if "track_buffer" not in cfg.tracker and sample_fps:
        merged["track_buffer"] = max(1, round(cfg.track_buffer_s * sample_fps))
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
    def __init__(
        self,
        cfg: DetectionConfig | None = None,
        per_camera: dict[str, DetectionConfig] | None = None,
    ):
        self.cfg = cfg or DetectionConfig()
        # Already-resolved effective configs for cameras that carry a
        # DetectionOverride (CLD-101), keyed by camera id. Resolved by
        # the caller (`DetectionConfig.for_camera`) rather than here, so
        # this module keeps knowing nothing about CameraConfig — it
        # receives plain DetectionConfigs either way.
        self._per_camera = per_camera or {}
        # One YOLO instance per camera: ultralytics keeps ByteTrack state
        # on the predictor, so sharing one instance across cameras would
        # tangle their track IDs. The nano model is small enough that a
        # per-camera copy is cheap, and each stays warm-loaded (the Ray
        # actor pattern, PRD §7) — which is also what makes a per-camera
        # `model` override just another entry here rather than a special
        # case.
        self._models: dict[str, Any] = {}
        # Keyed by (camera, sample_fps): the buffer length is derived
        # from the fps and the rest of the tracker dict may differ per
        # camera. Constant per camera, so the path handed to model.track
        # never changes under a live tracker. Distinct cameras with
        # identical settings hash to the same file on disk.
        self._tracker_paths: dict[tuple[str, float | None], Path] = {}

    def _cfg_for(self, camera_id: str) -> DetectionConfig:
        return self._per_camera.get(camera_id, self.cfg)

    def _tracker_for(self, camera_id: str, sample_fps: float | None) -> Path:
        key = (camera_id, sample_fps)
        path = self._tracker_paths.get(key)
        if path is None:
            path = tracker_config_path(self._cfg_for(camera_id), sample_fps)
            self._tracker_paths[key] = path
        return path

    def _model_for(self, camera_id: str):
        model = self._models.get(camera_id)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(self._cfg_for(camera_id).model)
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
        cfg = self._cfg_for(camera_id)
        model = self._model_for(camera_id)
        # Run the detector at the lowest applicable threshold; per-class
        # minimums are applied to its output below. (YOLO's conf is a
        # single global floor.)
        floor = min(
            [cfg.confidence, *cfg.class_confidence.values()]
        )
        results = model.track(
            image,
            persist=True,
            conf=floor,
            device=cfg.device,
            tracker=str(self._tracker_for(camera_id, payload.get("sample_fps"))),
            verbose=False,
        )

        h, w = image.shape[:2]
        zones = payload.get("zones") or []
        require_zone = bool(payload.get("require_zone")) and bool(zones)
        wanted = set(cfg.classes)

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
            if confidence < cfg.class_confidence.get(
                class_name, cfg.confidence
            ):
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            track_id = int(boxes.id[i]) if boxes.id is not None else None

            hit_zones = _zones_hit(zones, (x1, y1, x2, y2), w, h)
            if require_zone and not hit_zones:
                continue

            cx1, cy1, cx2, cy2 = expand_box(
                (x1, y1, x2, y2), cfg.crop_margin, w, h
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
