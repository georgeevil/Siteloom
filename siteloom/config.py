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


class EventRulesOverride(BaseModel):
    """Per-camera overrides for EventConfig — only non-None fields apply."""

    min_detections: int | None = None
    min_duration_s: float | None = None
    min_confidence: float | None = None
    stitch_gap_s: float | None = None


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
    # Per-camera overrides of the site-wide event rules (a busy doorway
    # and a quiet driveway want different stitching/significance gates).
    events: "EventRulesOverride | None" = None


class UniFiConfig(BaseModel):
    host: str = ""
    port: int = 443
    username: str = ""
    password: str = ""
    # Protect API key (Settings → Control Plane → Integrations). Optional:
    # the username/password pair covers everything the adapter does today;
    # the key additionally unlocks Protect's public API endpoints.
    api_key: str = ""
    verify_ssl: bool = False


class BackendConfig(BaseModel):
    # PRD §7: LocalBackend for the PoC; "celery"/"ray" reserved for later.
    kind: Literal["local"] = "local"


class DetectionConfig(BaseModel):
    model: str = "yolo11n.pt"
    device: str = "mps"
    confidence: float = 0.4
    # Tracker settings merged over ByteTrack defaults (keys as in
    # ultralytics' bytetrack.yaml). fuse_score is off by default because
    # we sample streams at a few fps, not native rate: fused cost is
    # 1 - IoU*conf, and a walking subject's between-sample IoU (~0.3)
    # times any realistic confidence always lands above the tracker's
    # hard-coded 0.7 new-track confirmation ceiling — so tracks are
    # created and instantly discarded, and every detection becomes its
    # own event (CLD-5).
    tracker: dict[str, float | int | bool | str] = {"fuse_score": False}
    # Context kept around each detection when the crop is cut, as a
    # fraction of the box's own width/height on each side (clamped to the
    # frame). A tight bbox crop is unreadable as a thumbnail and clips the
    # top of a head; a little surrounding frame makes it legible.
    # NOTE: the crop is both what gets stored and what the identity
    # embedders see, so changing this changes the embedding space —
    # re-enroll faces and `siteloom classes rebuild` after changing it.
    crop_margin: float = Field(0.12, ge=0.0, le=1.0)
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


class EventConfig(BaseModel):
    """Event consolidation and significance gating.

    A sampled stream fragments one physical visit into many short events
    (trackless detections, tracker restarts, class flapping). Stitching
    reattaches those fragments at ingest time; the significance gate keeps
    the fragments that remain out of the default triage view and away from
    identity resolution. Ephemeral (insignificant) events are stored and
    inspectable — never dropped.
    """

    # An event is significant once it has this many detections...
    min_detections: int = 3
    # ...spanning at least this long (0 = off; duration is largely
    # redundant with the count at a fixed sample rate)...
    min_duration_s: float = 0.0
    # ...with at least one detection this confident (gates on
    # Event.best_confidence; the detector's own threshold stays lower so
    # weak evidence is still recorded).
    min_confidence: float = 0.5
    # Stitch a trackless or fresh-track detection onto the most recent
    # event of the same camera + class group seen within this many
    # seconds, if their boxes overlap. Keep well under EVENT_LINK_GAP_S.
    stitch_gap_s: float = 15.0
    # Minimum IoU between the new detection and the candidate event's
    # last detection for the stitch to apply (guards against merging two
    # subjects crossing the same camera back-to-back).
    stitch_min_iou: float = 0.05
    # Classes the detector flaps between mid-track; members of a group
    # share events instead of splitting them.
    class_groups: list[list[str]] = [["car", "truck", "bus"]]
    # Identity resolution gates: skip crops from weak or tiny detections,
    # and (by default) skip events still below the significance gate —
    # each unresolved fragment otherwise mints a fresh unknown identity.
    identify_min_confidence: float = 0.5
    identify_min_crop_px: int = 48
    identify_only_significant: bool = True

    def for_camera(self, camera: "CameraConfig") -> "EventConfig":
        """Effective rules for one camera: site defaults + overrides.

        Shared by ingest and `siteloom events retag` so the two can never
        disagree about what "significant" means.
        """
        if camera.events is None:
            return self
        merged = self.model_copy()
        for field, value in camera.events.model_dump().items():
            if value is not None:
                setattr(merged, field, value)
        return merged

    def group_for(self, class_name: str) -> list[str]:
        """The class group containing class_name (or just itself)."""
        for group in self.class_groups:
            if class_name in group:
                return group
        return [class_name]


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
    # Required score gap between the best and second-best *identity*
    # before a visual match is accepted. Two identities inside this band
    # are an ambiguous read: the frame is left unresolved (after the
    # camera-recency tie-break) rather than guessed — one borderline
    # vector in the wrong gallery must not win outright (CLD-41).
    min_margin: float = 0.0
    # Consistency gate on minting unknown identities: a sub-threshold
    # embedding is parked in the identifier's pending pool and an
    # Identity row is only created once this many mutually-similar
    # sightings accumulate (Frigate's consistency principle). 1 = mint
    # immediately (the pre-CLD-41 behavior; auto-added classes keep it).
    min_sightings: int = 1
    # Detection confidence at or above which a single crop is trusted to
    # mint an identity immediately, bypassing min_sightings. A plate
    # always bypasses — plates are exact evidence.
    immediate_quality: float = 0.85


def _default_identifiers() -> dict[str, IdentifierConfig]:
    return {
        # People are the unknown-identity churn source (CLD-41): both
        # person identifiers demand two consistent sightings before an
        # unknown is minted, and a margin so near-ties stay unresolved
        # instead of guessed. Vehicles keep min_sightings=1 — they are
        # fewer, their visual threshold is already strict, and the
        # plate-learning flow (PRD §6.4) expects first-sighting rows.
        "face": IdentifierConfig(
            algo="face",
            applies_to=["person"],
            threshold=0.36,
            min_margin=0.05,
            min_sightings=2,
        ),
        "person": IdentifierConfig(
            algo="generic",
            applies_to=["person"],
            threshold=0.80,
            min_margin=0.02,
            min_sightings=2,
        ),
        "vehicle": IdentifierConfig(
            algo="generic",
            applies_to=["car", "truck", "bus", "motorcycle"],
            threshold=0.82,
            min_margin=0.02,
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
    # Camera-recency tie-break window (CLD-41): when two identities land
    # inside an identifier's min_margin band, one seen on the SAME camera
    # within this many seconds (stream time) wins the tie. Soft prior
    # only — it never overrides a clear better match, and losing it
    # (process restart) degrades to the conservative no-match.
    recency_window_s: float = 120.0
    # Pending-pool entries older than this are pruned: "consistent
    # sightings" means within this horizon, and a one-off blurry crop
    # must not linger as a promotion seed forever.
    pending_ttl_s: float = 3600.0


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
    events: EventConfig = EventConfig()
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
