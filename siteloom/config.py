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
    # face ID on a driveway-only camera). "identity" = second-pass
    # face/plate/re-ID on detection crops; "audio" = loud-duration
    # tracking on the stream's audio.
    modules: list[str] = ["detection", "identity"]
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


class IdentifierConfig(BaseModel):
    """One identification algorithm instance.

    Face ID algorithms are far more mature than generic person/vehicle
    re-ID, so each identifier picks its own `algo` and threshold rather
    than sharing one pipeline. Every identifier owns one vector-store
    collection (named by its registry key).
    """

    # "face": dedicated face pipeline (detector + alignment + face
    #         embedding — currently OpenCV YuNet + SFace).
    # "generic": appearance embedding from an ImageNet backbone; works
    #            for any object class, which is what makes dynamically
    #            added classes possible.
    algo: Literal["face", "generic"] = "generic"
    # Which detection classes this identifier consumes.
    applies_to: list[str]
    # Cosine-similarity threshold for "same identity". Face and generic
    # embeddings have very different similarity distributions — that is
    # the point of per-identifier thresholds.
    threshold: float = 0.80
    # Cap on embeddings stored per identity (label-and-learn keeps
    # improving matches up to this many samples).
    max_vectors_per_identity: int = 20
    # Vehicle path only: also try plate OCR on crops and match by plate
    # first (PRD §6.4). Requires the "plates" optional dependencies.
    plate_ocr: bool = False


def _default_identifiers() -> dict[str, IdentifierConfig]:
    return {
        "face": IdentifierConfig(algo="face", applies_to=["person"], threshold=0.36),
        "person": IdentifierConfig(algo="generic", applies_to=["person"], threshold=0.80),
        "vehicle": IdentifierConfig(
            algo="generic",
            applies_to=["car", "truck", "bus", "motorcycle"],
            threshold=0.82,
            plate_ocr=True,
        ),
    }


class IdentityConfig(BaseModel):
    enabled: bool = True
    # Qdrant local-mode directory (embedded engine, no server).
    vector_db_path: str = "identity_db"
    identifiers: dict[str, IdentifierConfig] = Field(
        default_factory=_default_identifiers
    )
    # Dynamically added classes: when a detection class has no configured
    # identifier, create a generic one for it at runtime (own collection,
    # default threshold) instead of ignoring it.
    auto_add_classes: bool = True
    auto_add_threshold: float = 0.80
    # Classes never auto-identified (too generic to re-identify usefully).
    auto_add_exclude: list[str] = ["bird"]
    # Fine-tuned face projection matrix (.npy) produced by
    # `siteloom train-face`. Empty = use raw SFace embeddings.
    face_projection_path: str = "training/face_projection.npy"


class AudioConfig(BaseModel):
    """Loud-duration tracking (PRD §6.5). Levels are dBFS, not SPL."""

    enabled: bool = True
    threshold_db: float = -25.0  # episode starts above this
    min_duration_s: float = 30.0  # ... and must last this long to count
    window_s: float = 1.0  # RMS window
    # An episode ends after this much continuous quiet.
    release_s: float = 5.0


class GuestConfig(BaseModel):
    """Booking correlation (PRD §6.7): iCal source + arrival window."""

    ical: str = ""  # URL or local .ics path
    # Events within [checkin - pre, checkin + post] hours are stamped
    # guest_window=True to suppress unknown-vehicle alarms.
    arrival_pre_hours: float = 2.0
    arrival_post_hours: float = 4.0


class LibraryConfig(BaseModel):
    """Local media library indexing (photos / short videos)."""

    # Frames sampled per video during indexing — enough to catch everyone
    # present without processing an archive frame by frame.
    video_frames: int = 5
    # Default batch size for the resumable process phase.
    batch_size: int = 100
    # Run identification during indexing (can be deferred to save time
    # on a first pass, then run later).
    identify_on_index: bool = True


class TrainingConfig(BaseModel):
    """Face model training from verified library annotations."""

    output_dir: str = "training"
    # Face embedder fine-tuning: a projection head over SFace features.
    min_samples_per_person: int = 5
    embed_epochs: int = 60
    embed_lr: float = 1e-3
    embed_output_dim: int = 128
    # Fraction of people's samples held out for evaluation.
    val_fraction: float = 0.25
    # YOLO face detector training.
    detector_model: str = "yolo11n.pt"
    detector_epochs: int = 50
    detector_imgsz: int = 640


class MqttConfig(BaseModel):
    """MQTT connection, used both to publish Siteloom events and to
    consume other systems' (Frigate's) event streams."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    client_id: str = "siteloom"
    # Root for published topics: <base_topic>/events, <base_topic>/identity.
    base_topic: str = "siteloom"


class FrigateConfig(BaseModel):
    """Consume an existing Frigate install's events.

    Frigate keeps doing what it is good at (RTSP ingest, motion gating,
    first-pass object detection); Siteloom takes over the recognition
    layer — the role Double Take + CompreFace play in that stack.
    """

    enabled: bool = False
    # Frigate's HTTP API, used to fetch event snapshots.
    api_url: str = "http://localhost:5000"
    mqtt_topic: str = "frigate/events"
    # Only these Frigate labels are processed (empty = all).
    labels: list[str] = ["person", "car", "truck", "motorcycle", "bus"]
    # Only these cameras (empty = all).
    cameras: list[str] = []
    min_score: float = 0.6
    # Frigate fires "update" events as a tracked object improves; process
    # at most one snapshot per event this many seconds apart.
    update_interval_s: float = 10.0


class WebhookConfig(BaseModel):
    url: str
    # Which occurrences fire this hook: identity.match, identity.unknown,
    # identity.new_plate, noise.episode.
    events: list[str] = ["identity.match", "identity.unknown"]
    # Optional bearer token sent as Authorization: Bearer <token>.
    token: str = ""


class RecognitionApiConfig(BaseModel):
    """CompreFace-compatible face recognition REST API.

    Lets tools that already speak CompreFace (Double Take most notably)
    use Siteloom as their recognizer — same face collection the cameras
    and the photo backfill share.
    """

    enabled: bool = True
    # If set, requests must carry it in the x-api-key header (the header
    # CompreFace clients already send).
    api_key: str = ""
    # Minimum face-detector confidence for a box to be reported.
    det_prob_threshold: float = 0.7


class IntegrationsConfig(BaseModel):
    mqtt: MqttConfig = MqttConfig()
    frigate: FrigateConfig = FrigateConfig()
    recognition_api: RecognitionApiConfig = RecognitionApiConfig()
    webhooks: list[WebhookConfig] = []


class StorageConfig(BaseModel):
    db_url: str = "sqlite:///siteloom.db"
    # Directory where detection crops / annotated frames are written.
    media_dir: str = "media"


class SiteConfig(BaseModel):
    site_id: str
    site_name: str = ""
    # Empty is valid: a library-only deployment (archive indexing and
    # labeling) needs no cameras.
    cameras: list[CameraConfig] = []
    unifi: UniFiConfig = UniFiConfig()
    backend: BackendConfig = BackendConfig()
    detection: DetectionConfig = DetectionConfig()
    identity: IdentityConfig = IdentityConfig()
    audio: AudioConfig = AudioConfig()
    guests: GuestConfig = GuestConfig()
    library: LibraryConfig = LibraryConfig()
    training: TrainingConfig = TrainingConfig()
    integrations: IntegrationsConfig = IntegrationsConfig()
    storage: StorageConfig = StorageConfig()


def load_config(path: str | Path) -> SiteConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    config = SiteConfig.model_validate(data)
    # Remembered so operator edits made in the web UI (class list,
    # thresholds) can be written back to the file they came from.
    object.__setattr__(config, "_source_path", str(Path(path).resolve()))
    return config


def save_config(config: SiteConfig, path: str | Path | None = None) -> str:
    target = path or getattr(config, "_source_path", None)
    if not target:
        raise ValueError("no config path known; pass one explicitly")
    data = config.model_dump(mode="json")
    Path(target).write_text(yaml.safe_dump(data, sort_keys=False))
    return str(target)
