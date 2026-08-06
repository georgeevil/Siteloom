"""Site configuration: pydantic models + YAML loader.

Per NFR3 the system is config-driven: each camera is a YAML entry declaring
its adapter, stream source, polygon zones, and which processing modules run
on it — never a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ZoneConfig(BaseModel):
    """A named polygon zone in camera coordinates (normalized 0..1)."""

    name: str
    points: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("points")
    @classmethod
    def _normalized(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("zone points must be normalized to 0..1")
        return pts


class CameraConfig(BaseModel):
    id: str
    name: str = ""
    adapter: Literal["unifi", "file", "rtsp"] = "unifi"
    # Adapter-specific source: UniFi camera id, file/dir path, or RTSP URL.
    source: str
    # Which processing modules run on this camera (PRD §6.1: e.g. skip
    # face ID on a driveway-only camera). The vertical slice only ships
    # "detection"; future modules subscribe by name.
    modules: list[str] = ["detection"]
    # Frames per second sampled from the stream for processing.
    sample_fps: float = 2.0
    zones: list[ZoneConfig] = []
    # Only report detections inside at least one zone. If no zones are
    # configured the full frame is used regardless of this flag.
    require_zone: bool = False


class UniFiConfig(BaseModel):
    host: str = ""
    port: int = 443
    username: str = ""
    password: str = ""
    verify_ssl: bool = False


class BackendConfig(BaseModel):
    # PRD §7: LocalBackend for the PoC; "celery"/"ray" reserved for later.
    kind: Literal["local"] = "local"


class DetectionConfig(BaseModel):
    model: str = "yolo11n.pt"
    device: str = "mps"
    confidence: float = 0.4
    # COCO class names we care about (PRD §6.2).
    classes: list[str] = [
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "dog",
        "cat",
        "bird",
    ]


class StorageConfig(BaseModel):
    db_url: str = "sqlite:///siteloom.db"
    # Directory where detection crops / annotated frames are written.
    media_dir: str = "media"


class SiteConfig(BaseModel):
    site_id: str
    site_name: str = ""
    cameras: list[CameraConfig]
    unifi: UniFiConfig = UniFiConfig()
    backend: BackendConfig = BackendConfig()
    detection: DetectionConfig = DetectionConfig()
    storage: StorageConfig = StorageConfig()


def load_config(path: str | Path) -> SiteConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SiteConfig.model_validate(data)
