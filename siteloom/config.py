"""Site configuration: pydantic models + YAML loader.

Per NFR3 the system is config-driven: each camera is a YAML entry declaring
its adapter, stream source, polygon zones, and which processing modules run
on it — never a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Per-camera overrides for EventConfig — only non-None fields apply.

    Every scalar gate on EventConfig is overridable: a busy doorway and a
    quiet driveway disagree about what counts as significant *and* about
    which crops are worth identifying, and the identify gates are where
    the unknown-identity churn is actually controlled (CLD-39). Structural
    fields (`class_groups`) stay site-wide — they describe the detector's
    class flapping, not the camera.
    """

    min_detections: int | None = None
    min_duration_s: float | None = None
    min_confidence: float | None = None
    stitch_gap_s: float | None = None
    stitch_min_iou: float | None = None
    stitch_candidates: int | None = None
    track_link_gap_s: float | None = None
    merge_gap_s: float | None = None
    identify_min_confidence: float | None = None
    identify_min_crop_px: int | None = None
    identify_only_significant: bool | None = None


class PlateFloors(NamedTuple):
    """The plate-quality floors in force for one read (CLD-128).

    One value object rather than four loose keywords, so the resolution
    (`IdentityConfig.plate_floors_for`) has a shape ingest and any future
    replay share — the same reason `threshold_for` exists. `_asdict()`
    is the serializable form that rides in the identity job payload.
    """

    min_chars: int
    min_width_px: int
    min_sharpness: float
    min_char_confidence: float


class PlateFloorsOverride(BaseModel):
    """Per-camera plate-quality floors (CLD-128) — only set fields apply.

    Legibility is a property of *this camera at this distance in this
    light*, not of the site: 100 px is a sane width floor on a gate
    camera at two metres and rejects every correct read on a camera
    watching the street (measured on `backyard-puerta`: correct reads
    run 34–100 px there, and six reads in the whole table ever exceeded
    100 px). A field left None inherits the identifier's site-wide
    value, exactly like `EventRulesOverride`.
    """

    min_chars: int | None = None
    min_width_px: int | None = None
    min_sharpness: float | None = None
    min_char_confidence: float | None = None

    @field_validator("min_chars", "min_width_px")
    @classmethod
    def _non_negative_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("plate floors cannot be negative")
        return value

    @field_validator("min_sharpness")
    @classmethod
    def _non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("plate floors cannot be negative")
        return value

    @field_validator("min_char_confidence")
    @classmethod
    def _confidence_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(
                "plate_floors.min_char_confidence is a probability in 0..1"
            )
        return value


class CameraIdentityOverride(BaseModel):
    """Per-camera identity gates (CLD-39).

    `thresholds` is keyed by *identifier* (face/person/vehicle/an
    auto-added class), never by algorithm or by camera-wide scalar: each
    identifier's cosine similarity distribution is its own world (face
    lives around 0.36, generic around 0.80), so one number per camera
    would be meaningless. A key absent here follows the site-wide
    `IdentifierConfig.threshold`; a key with no configured identifier is
    still honoured, because `auto_add_classes` mints identifiers at
    runtime that the YAML never named.
    """

    thresholds: dict[str, float] = {}
    # Per-camera plate-quality floors (CLD-128). Flat, not keyed by
    # identifier: the floors describe what this camera's pixels can
    # carry, which is the same fact for every identifier reading them.
    plate_floors: PlateFloorsOverride | None = None

    @field_validator("thresholds")
    @classmethod
    def _in_range(cls, values: dict[str, float]) -> dict[str, float]:
        for key, value in values.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"identity threshold for {key!r} must be a cosine "
                    "similarity in 0..1"
                )
        return values


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
    # Per-camera identity gates — per-identifier similarity thresholds.
    identity: "CameraIdentityOverride | None" = None


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
    # Ceiling on one clip export (`get_camera_video`). The NVR has been
    # observed going silent mid-transfer with the TCP connection left
    # ESTABLISHED — without a deadline the read blocks forever and the
    # backfill sits "running" with a cold heartbeat (a 69-minute stall in
    # practice, on a clip that takes seconds). Generous on purpose: it
    # exists to catch a dead transfer, never to cut off a slow one — a
    # chunked backfill window of several minutes of 4K video is still far
    # inside it. 0 disables the deadline.
    download_timeout_s: float = 120.0


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
    # Per-class minimum confidence, overriding `confidence` for that
    # class (e.g. demand more of "dog" than "person"). The detector runs
    # at the lowest applicable threshold and per-class filtering happens
    # on its output, so a class absent here uses `confidence`.
    class_confidence: dict[str, float] = {}
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
    # A detection joins its track's existing event only if that event was
    # last seen this recently (frame time, symmetric for backfill order).
    # Track ids restart at 1 whenever a tracker is rebuilt — process
    # restart, stream reconnect, the next backfill clip — so track id
    # alone would staple today's visitor onto last week's event.
    track_link_gap_s: float = 120.0
    # Stitch a trackless or fresh-track detection onto a recent event of
    # the same camera + class group seen within this many seconds, if
    # their boxes overlap. Keep well under track_link_gap_s.
    stitch_gap_s: float = 15.0
    # Minimum IoU between the new detection and the candidate event's
    # last detection for the stitch to apply (guards against merging two
    # subjects crossing the same camera back-to-back).
    stitch_min_iou: float = 0.05
    # How many recent candidate events the stitcher tries, newest first.
    # One is not enough: with two subjects in frame the newest event is
    # usually the *other* subject, the IoU guard rejects it, and every
    # frame mints a fresh fragment (CLD-40).
    stitch_candidates: int = 5
    # After a detection resolves to an identity, fold its event into a
    # recent event on the same camera already linked to that identity —
    # the fragments stitching cannot catch because the subject moved too
    # far between samples. 0 disables. Frame time, symmetric.
    merge_gap_s: float = 60.0
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
    # Shortest normalized OCR read accepted as a plate. Configuration
    # rather than a literal because it is the exact bar short/angled
    # motorcycle plates fall under (CLD-9), and the answer to "is the
    # floor in the right place" is "move it and re-read the table".
    # Reads under it are still recorded — a `PlateRead` row with reason
    # "too-short" keeps the raw text — so lowering it is a question about
    # existing data, not a reason to re-run anything.
    plate_min_chars: int = 4
    # Image-quality floors on a plate read, all 0 = off. OCR confidence
    # is a poor rejection signal on its own — a smeared 60-pixel plate
    # comes back at 0.9 because the network is confident about the
    # characters it hallucinated — so what gets rejected is measured off
    # the image instead of asked of the model:
    #
    #   plate_min_width_px      the plate region's width in source
    #                           pixels. The closest thing LPR has to a
    #                           hardware spec: under ~100 px the
    #                           characters carry too few pixels to be
    #                           distinguished. Start around 90-110.
    #   plate_min_sharpness     variance of the Laplacian over the plate
    #                           region — the standard blur measure, and
    #                           the one that separates "small but crisp"
    #                           from "large and motion-smeared". Scale-
    #                           and exposure-dependent, so calibrate it
    #                           off your own reads on /plates (the value
    #                           is on every row) rather than copying a
    #                           number from another install.
    #   plate_min_char_confidence
    #                           floor on the *weakest* character's
    #                           probability, not the mean. The mean is
    #                           what hides a single substituted
    #                           character: five characters at 0.98 and
    #                           one at 0.35 average to 0.87. Somewhere
    #                           around 0.5-0.6 is a starting point.
    #
    # Every read is measured and recorded whether or not a floor is set,
    # and a read that fails one is a row with a reason — so the floors
    # are chosen by reading the table, and moving one never requires
    # re-running anything. A metric the OCR never reported cannot fail a
    # floor: absent is absent, not zero.
    plate_min_width_px: int = 0
    plate_min_sharpness: float = 0.0
    plate_min_char_confidence: float = 0.0
    # Floor on how often the same vehicle is OCR'd, in seconds of frame
    # time (CLD-130). A car dwelling in frame is OCR'd on every sampled
    # frame otherwise — one parked car produced 1,437 reads in six
    # minutes, so the screen an operator opens to ask "which plates came
    # past?" was a per-frame log of one vehicle, and most of the
    # inference budget re-derived a string it already had. A *time* cap,
    # never a count cap: cross-frame consensus (CLD-114) wants several
    # independent reads spread over a visit, and a count cap would
    # starve a long approach of exactly those samples. 0 = every frame
    # (the pre-CLD-130 behavior). The embedding still runs on skipped
    # frames — only the OCR is rationed.
    plate_ocr_interval_s: float = 1.0
    # Keep the plate sub-crop for each read, under `<media_dir>/plates/`.
    # A third image with its own purpose: `crop_jpeg` is simultaneously
    # the display thumbnail and the embedder input, so the evidence image
    # for an OCR read must not touch it.
    plate_save_crops: bool = True
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
    # Learning gates (CLD-139). CLD-41's gates guard *minting*; these
    # guard accretion, which was unconditional: every matching frame
    # added its embedding, so one 30-frame visit could fill a gallery
    # and one wrong match became a permanent attractor that recruited
    # more wrong vectors on every later visit.
    #
    #   learn_min_quality   detection confidence at or below which a
    #                       frame is matched but never *stored*. The
    #                       only quality signal the resolver receives
    #                       today is the detector's own confidence in
    #                       the parent box — it says "this is a person",
    #                       not "this crop is legible" — so this is a
    #                       floor on the signal available, not the ideal
    #                       one (sharpness/size, the way the plate
    #                       floors do it, is the follow-up). A frame
    #                       that reports no quality at all passes:
    #                       absent is absent, not zero, exactly as for
    #                       the plate floors above. It applies from an
    #                       identity's *second* vector: the pending pool
    #                       does not carry a sighting's quality, and a
    #                       floored promotion could found an empty
    #                       gallery — invisible to matching, so the next
    #                       frame mints yet another identity.
    #   learn_max_per_event how many vectors one event may contribute to
    #                       one identity. The cap that makes a single
    #                       visit unable to flood a gallery; the
    #                       per-identity cap (max_vectors_per_identity)
    #                       still applies on top, and the lower of the
    #                       two wins.
    #
    # 0 switches either gate off, restoring the pre-CLD-139 behaviour
    # without a redeploy — which is this change's rollback lever.
    learn_min_quality: float = 0.6
    learn_max_per_event: int = 3


class FingerprintConfig(BaseModel):
    """Flock-style vehicle attributes (CLD-254), off by default.

    Color is pure pixel math on the crop the pipeline already carries —
    no model — so the only cost of turning it on is a few columns per
    detection. Body type comes free from the YOLO class and plate
    status from existing PlateRead rows; color is the one new
    measurement, and its floors follow the plate-floor discipline:
    every read records its measurements next to the floor applied, so
    moving a floor is a question about existing data.
    """

    enabled: bool = False
    # Detection classes fingerprinted. Matches the vehicle identifier's
    # `applies_to` by default; a class outside this list is never
    # measured, flag or no flag.
    classes: list[str] = ["car", "truck", "bus", "motorcycle"]
    # Crops narrower than this on either side name no color — the
    # center-region vote over a handful of pixels is noise, the same
    # reason plates have a width floor.
    min_px: int = 32
    # 95th-percentile per-pixel channel spread below which the crop is
    # achromatic and no color is named. This is the IR-honesty floor: a
    # grayscale frame must read "unknown (IR)", never a confidently
    # wrong "gray". Measured over the whole crop, margin included,
    # because daylight background is what separates a white car from an
    # IR frame.
    chroma_floor: float = 12.0


def _default_identifiers() -> dict[str, IdentifierConfig]:
    return {
        # People are the unknown-identity churn source (CLD-41): both
        # person identifiers demand two consistent sightings before an
        # unknown is minted, and a margin so near-ties stay unresolved
        # instead of guessed. Vehicles keep min_sightings=1 — they are
        # fewer, their visual threshold is already strict, and the
        # plate-learning flow (PRD §6.4) expects first-sighting rows.
        #
        # The learn gates (CLD-139) are equal across all three on
        # purpose: `quality` is the YOLO confidence of the one parent
        # box, so face and person receive the *identical* number for the
        # same crop. The per-identifier knob exists so a site can tighten
        # one doorway, not because the signal differs. Stated here rather
        # than inherited from the field defaults so `doctor` and a reader
        # of this file see the values in force.
        "face": IdentifierConfig(
            algo="face",
            applies_to=["person"],
            threshold=0.36,
            min_margin=0.05,
            min_sightings=2,
            learn_min_quality=0.6,
            learn_max_per_event=3,
        ),
        "person": IdentifierConfig(
            algo="generic",
            applies_to=["person"],
            threshold=0.80,
            min_margin=0.02,
            min_sightings=2,
            learn_min_quality=0.6,
            learn_max_per_event=3,
        ),
        "vehicle": IdentifierConfig(
            algo="generic",
            applies_to=["car", "truck", "bus", "motorcycle"],
            threshold=0.82,
            min_margin=0.02,
            plate_ocr=True,
            learn_min_quality=0.6,
            learn_max_per_event=3,
        ),
    }


class IdentityConfig(BaseModel):
    enabled: bool = True
    # Qdrant local-mode directory (embedded engine, no server).
    vector_db_path: str = "identity_db"
    identifiers: dict[str, IdentifierConfig] = Field(
        default_factory=_default_identifiers
    )
    # Vehicle fingerprint attributes (CLD-254). A plain sub-model, not
    # part of the identifier overlay below: it has no per-key built-ins
    # to merge, so partial YAML already keeps the other defaults.
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)

    @model_validator(mode="before")
    @classmethod
    def _overlay_identifier_defaults(cls, data):
        """A named identifier overlays its built-in defaults (CLD-125).

        `identifiers` is a dict with a default_factory, so a config that
        spells out its identifiers used to REPLACE `_default_identifiers`
        wholesale: every field the operator did not restate fell back to
        the bare field defaults, which for `min_margin`/`min_sightings`
        are 0.0/1 — the pre-CLD-41 behavior. Naming `face:` only to
        adjust its `threshold:` therefore switched off consistency
        gating, silently, and the more carefully a site was tuned the
        more of CLD-41 it lost. That is what the live site ran with.

        So an entry whose key is a built-in one is merged *over* the
        built-in rather than replacing it, and only for keys the operator
        named — omitting `vehicle` entirely still removes it, which is
        the one thing wholesale replacement got right.

        This runs `mode="before"`, on the raw mapping, for the reason
        that makes the whole fix work: after validation an absent
        `min_sightings` and an explicit `min_sightings: 1` are the same
        value, and an operator who deliberately wants first-sighting
        minting must still be able to say so.
        """
        if not isinstance(data, dict):
            return data
        given = data.get("identifiers")
        if not isinstance(given, dict):
            return data

        builtin = _default_identifiers()
        merged: dict[str, object] = {}
        for key, entry in given.items():
            base = builtin.get(key)
            if base is None or entry is None:
                merged[key] = entry
                continue
            # Entries arrive as mappings from YAML and as models from
            # code (tests, the console's config editor); both have to
            # merge, and a model was built with every field set, so it
            # carries no "unstated" fields to fill in.
            if isinstance(entry, IdentifierConfig):
                merged[key] = entry
                continue
            if not isinstance(entry, dict):
                merged[key] = entry
                continue
            merged[key] = {**base.model_dump(), **entry}

        return {**data, "identifiers": merged}
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

    def threshold_for(
        self, identifier_key: str, camera: "CameraConfig | None" = None
    ) -> float | None:
        """Effective cosine threshold for one identifier on one camera.

        Camera override first, then the identifier's site-wide value.
        None means "nothing configured here" — the resolver then falls
        back to the identifier config the registry holds, which is where
        an auto-added class's default lives. The counterpart of
        `EventConfig.for_camera`: one place decides, so ingest and any
        future replay can never disagree.
        """
        if camera is not None and camera.identity is not None:
            value = camera.identity.thresholds.get(identifier_key)
            if value is not None:
                return value
        ident = self.identifiers.get(identifier_key)
        return ident.threshold if ident is not None else None

    def plate_floors_for(
        self, identifier_key: str, camera: "CameraConfig | None" = None
    ) -> PlateFloors:
        """Effective plate-quality floors for one identifier on one camera.

        Camera override field by field, then the identifier's site-wide
        values, then the bare defaults (an identifier the config never
        named — the registry's auto-added kind — has no plate OCR, but
        asking is still answerable). The counterpart of `threshold_for`
        one floor down (CLD-128): one place decides, so ingest and any
        future replay cannot disagree about which bar a read faced.
        """
        ident = self.identifiers.get(identifier_key)
        floors = PlateFloors(
            min_chars=ident.plate_min_chars if ident is not None else 4,
            min_width_px=ident.plate_min_width_px if ident is not None else 0,
            min_sharpness=ident.plate_min_sharpness if ident is not None else 0.0,
            min_char_confidence=(
                ident.plate_min_char_confidence if ident is not None else 0.0
            ),
        )
        override = (
            camera.identity.plate_floors
            if camera is not None and camera.identity is not None
            else None
        )
        if override is None:
            return floors
        return PlateFloors(
            min_chars=(
                override.min_chars
                if override.min_chars is not None
                else floors.min_chars
            ),
            min_width_px=(
                override.min_width_px
                if override.min_width_px is not None
                else floors.min_width_px
            ),
            min_sharpness=(
                override.min_sharpness
                if override.min_sharpness is not None
                else floors.min_sharpness
            ),
            min_char_confidence=(
                override.min_char_confidence
                if override.min_char_confidence is not None
                else floors.min_char_confidence
            ),
        )


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
    # Directories the web import wizard (CLD-27) may register sources
    # under. Empty is the safe default and disables web import entirely —
    # the CLI still works, because someone with a shell already has the
    # filesystem. The wizard exists so an operator does not need one, and
    # that is exactly why it must not hand them an arbitrary-path read of
    # the host. Paths are resolved and compared with is_relative_to, the
    # same containment the media route uses.
    import_roots: list[str] = []


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
    # identity.new_plate, noise.episode, plate.watchlist.
    events: list[str] = ["identity.match", "identity.unknown"]
    # Optional bearer token sent as Authorization: Bearer <token>.
    token: str = ""


class RecognitionApiConfig(BaseModel):
    """CompreFace-compatible face recognition REST API.

    Lets tools that already speak CompreFace (Double Take most notably)
    use Siteloom as their recognizer — same face collection the cameras
    and the photo backfill share.

    This is a biometric surface (matching AND enrollment), so it is off
    by default (NFR5), and enabling it without an api_key refuses to
    start unless allow_open says the exposure is deliberate (CLD-47).
    """

    enabled: bool = False
    # If set, requests must carry it in the x-api-key header (the header
    # CompreFace clients already send).
    api_key: str = ""
    # Explicit opt-in to serve WITHOUT authentication. Enabled with no
    # api_key and no allow_open fails fast at startup rather than
    # exposing face matching and gallery writes to anyone on the port.
    allow_open: bool = False
    # Per-client-IP request budget over a sliding minute for /api/v1/;
    # 0 disables rate limiting.
    rate_limit_per_minute: int = 60
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
    # A relative value is relative to the config file, not the CWD — see
    # _ANCHORED_PATHS below.
    media_dir: str = "media"


class ServiceConfig(BaseModel):
    """How this deployment runs as a service (`siteloom service`).

    These live in the config rather than in `service install` flags for
    one reason: a value computed at install time freezes *this machine*
    into the generated unit, and the operator who copies site.yaml to the
    next host gets it wrong. `log_dir` is anchored like every other path
    (see _ANCHORED_PATHS), so the YAML keeps saying `logs` while the unit
    says the absolute path this host resolves it to.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    # Where the rotating log and (on launchd, which has no journal) the
    # crash streams go. Relative to the config file.
    log_dir: str = "logs"
    # Long enough for an in-flight batch to commit — see the batch-timing
    # note in docs/operations.md. Units raise this to 60 s for `run` and
    # `frigate`, which drain per-camera work rather than a single loop.
    stop_timeout_s: int = 30
    restart: str = "on-failure"  # on-failure | always | never
    restart_delay_s: int = 10
    start_at_boot: bool = True


class SiteConfig(BaseModel):
    site_id: str
    site_name: str = ""
    # The site's IANA timezone (CLD-100). Storage stays naive UTC by
    # contract; this is the zone every console screen and export converts
    # to at display, and the frame operator-typed `datetime-local` input
    # is read in. Empty = unset = UTC, labelled as UTC. Set from the
    # /classes admin panel: typed by an admin, detected from the UniFi
    # NVR, or seeded once from an admin's browser — `timezone_source`
    # records which rung supplied it ("admin" / "nvr" / "browser").
    timezone: str = ""
    timezone_source: Literal["", "admin", "nvr", "browser"] = ""
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
    service: ServiceConfig = ServiceConfig()

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        # A typo must be refused at the boundary, not stored: a bad name
        # in YAML otherwise passes load and dies at first render.
        from siteloom.localtime import validate_timezone

        return validate_timezone(value)


# Filesystem paths a site config can name. A relative one is anchored to
# the config file's own directory at load time, never to the process CWD
# (CLD-64): `siteloom serve` started by a service manager from /, a
# backfill run by hand from the repo root and a cron job launched from
# $HOME must all mean the same `media_dir`. Otherwise the same YAML
# writes crops in one tree and serves them from another — and because
# the /media route's containment check is anchored on media_dir too, a
# wrong root is a security-relevant answer, not just a 404.
#
# Anchoring happens once, in load_config, so that every consumer of the
# config (ingest, the library indexer, the Frigate snapshot writer,
# `doctor`, the web layer) inherits it without resolving anything itself.
# `storage.db_url` is deliberately NOT here: it is a URL, not a path, and
# a relative sqlite:/// URL still follows the CWD.
_ANCHORED_PATHS: tuple[tuple[str, str], ...] = (
    ("storage", "media_dir"),
    ("identity", "vector_db_path"),
    ("identity", "face_projection_path"),
    ("training", "output_dir"),
    ("service", "log_dir"),
)


def anchor_path(value: str, base: str | Path) -> Path:
    """`value` as an absolute path, relative values taken from `base`.

    `~` is expanded first: "~/media" is absolute to the operator who
    typed it, and joining it onto the config directory would be wrong.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(base) / path


def load_config(path: str | Path) -> SiteConfig:
    source = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    config = SiteConfig.model_validate(data)
    # Remembered so operator edits made in the web UI (class list,
    # thresholds) can be written back to the file they came from.
    object.__setattr__(config, "_source_path", str(source))
    raw: dict[tuple[str, str], str] = {}
    for section, field in _ANCHORED_PATHS:
        target = getattr(config, section)
        value = getattr(target, field)
        if not value:
            continue
        raw[(section, field)] = value
        setattr(target, field, str(anchor_path(value, source.parent)))
    # What the file actually said, so save_config can put it back: a
    # console save must not freeze this machine's absolute paths into a
    # config that gets copied to the next host.
    object.__setattr__(config, "_raw_paths", raw)
    return config


def save_config(config: SiteConfig, path: str | Path | None = None) -> str:
    target = path or getattr(config, "_source_path", None)
    if not target:
        raise ValueError("no config path known; pass one explicitly")
    data = config.model_dump(mode="json")
    source = getattr(config, "_source_path", None)
    anchored = getattr(config, "_raw_paths", None) or {}
    for (section, field), original in anchored.items():
        # Only restore a value nothing has changed since it was loaded —
        # a path the operator edited must be written as it now stands.
        current = getattr(getattr(config, section), field)
        if source and current == str(anchor_path(original, Path(source).parent)):
            data[section][field] = original
    Path(target).write_text(yaml.safe_dump(data, sort_keys=False))
    return str(target)
