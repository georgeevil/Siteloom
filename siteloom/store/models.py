"""Event store models (PRD §6.7, relational half).

An Event is one tracked object's visit: the same track ID on the same
camera, spanning first_seen..last_seen, with N individual Detections.

Identity rows are the relational half of the identity store: labels,
plates, appearance stats. The embeddings themselves live in the vector
store (Qdrant local mode, see siteloom/identity/vectors.py); the two are
joined by Identity.id, which is used as the vector point id's payload.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, default="")
    adapter: Mapped[str] = mapped_column(String, default="")

    events: Mapped[list["Event"]] = relationship(back_populates="camera")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Id of the originating event in an external system (Frigate event id)
    # — the dedupe key when consuming other NVRs' event streams.
    external_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    class_name: Mapped[str] = mapped_column(String, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    # Highest-confidence crop for this event — the thumbnail.
    best_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    best_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # True if the event falls inside a known guest arrival window
    # (booking correlation, PRD §6.7) — used to suppress false alarms.
    guest_window: Mapped[bool] = mapped_column(Boolean, default=False)

    camera: Mapped[Camera] = relationship(back_populates="events")
    detections: Mapped[list["Detection"]] = relationship(back_populates="event")
    identities: Mapped[list["EventIdentity"]] = relationship(back_populates="event")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    class_name: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[str] = mapped_column(Text)  # JSON [x1, y1, x2, y2]
    zones: Mapped[str] = mapped_column(Text, default="[]")  # JSON [name, ...]
    crop_path: Mapped[str | None] = mapped_column(String, nullable=True)

    event: Mapped[Event] = relationship(back_populates="detections")


class Identity(Base):
    """One recognized individual thing: a face, a person's appearance, a
    vehicle. `identifier_key` names which identification algorithm owns it
    ("face", "person", "vehicle", or a dynamically added class), so the
    same physical person can have both a face identity and an appearance
    identity — cross-linking them is a V1 concern.

    Label-and-learn (PRD §6.3): rows start with label=None and show up as
    "unknown-<id>" in the UI until the operator names them.
    """

    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identifier_key: Mapped[str] = mapped_column(String, index=True)
    class_name: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # Vehicle identities can also be matched by plate (PRD §6.4): plate OR
    # visual signature write to this same record, whichever is available.
    plate: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    appearance_count: Mapped[int] = mapped_column(Integer, default=0)
    # How many embeddings this identity has in the vector store (capped).
    vector_count: Mapped[int] = mapped_column(Integer, default=0)
    best_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)

    events: Mapped[list["EventIdentity"]] = relationship(back_populates="identity")

    @property
    def display_name(self) -> str:
        return self.label or f"unknown-{self.identifier_key}-{self.id}"


class EventIdentity(Base):
    """Links an Event to the identities recognized during it."""

    __tablename__ = "event_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("identities.id"), index=True)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    hit_count: Mapped[int] = mapped_column(Integer, default=1)

    event: Mapped[Event] = relationship(back_populates="identities")
    identity: Mapped[Identity] = relationship(back_populates="events")


class NoiseEvent(Base):
    """A sustained loud episode (PRD §6.5, NoiseAware/Minut model).

    Levels are dBFS (relative to digital full scale), not SPL — thresholds
    are calibrated per microphone/deployment, not absolute loudness.
    """

    __tablename__ = "noise_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime)
    peak_db: Mapped[float] = mapped_column(Float)
    mean_db: Mapped[float] = mapped_column(Float)


class Booking(Base):
    """A guest booking synced from iCal (PRD §6.7 guest-correlation)."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String, unique=True, index=True)
    summary: Mapped[str] = mapped_column(String, default="")
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime, index=True)


# --------------------------------------------------------------------------
# Media library: local directories of photos/short videos, indexed and
# labeled independently of live camera events. This is the training-data
# and archive side of the system; it shares the Identity store with live
# ingestion so a face enrolled from a photo archive is recognized on a
# camera immediately.
# --------------------------------------------------------------------------


class LibrarySource(Base):
    """A local directory registered for indexing."""

    __tablename__ = "library_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    path: Mapped[str] = mapped_column(String, unique=True, index=True)
    kind: Mapped[str] = mapped_column(String, default="directory")  # | "takeout"
    added_at: Mapped[datetime] = mapped_column(DateTime)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    items: Mapped[list["LibraryItem"]] = relationship(back_populates="source")


class LibraryItem(Base):
    """One media file. Indexing is resumable and partial by design: rows
    are created on scan (status="pending") and only processed when a
    detection pass reaches them, so a huge archive can be indexed in
    chunks across many runs without losing place.
    """

    __tablename__ = "library_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("library_sources.id"), index=True)
    path: Mapped[str] = mapped_column(String, unique=True, index=True)
    kind: Mapped[str] = mapped_column(String)  # "image" | "video"
    # pending -> indexed | failed | skipped
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mtime: Mapped[datetime] = mapped_column(DateTime)
    # Capture time from sidecar metadata (Takeout photoTakenTime) when
    # available — more trustworthy than mtime for archives.
    taken_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Human review state for the item as a whole, independent of whether
    # its individual annotations are verified.
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    source: Mapped[LibrarySource] = relationship(back_populates="items")
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )
    tags: Mapped[list["ItemTag"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )

    @property
    def name(self) -> str:
        from pathlib import Path

        return Path(self.path).name


class ItemTag(Base):
    """A whole-image tag. Namespaced by kind so imported metadata and
    operator tags coexist: kind="person" holds Google Photos people tags
    (the training signal), kind="user" holds free-form operator tags.
    """

    __tablename__ = "item_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("library_items.id"), index=True)
    kind: Mapped[str] = mapped_column(String, default="user", index=True)
    value: Mapped[str] = mapped_column(String, index=True)

    item: Mapped[LibraryItem] = relationship(back_populates="tags")


class Annotation(Base):
    """A box on a library item — machine-detected or human-drawn.

    One table serves detection review, identity labeling, custom-class
    labeling, and face-training data. `source` records provenance and
    `verified` records human sign-off; training exports only ever read
    verified rows, which is what keeps auto-assignments from silently
    becoming ground truth.
    """

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("library_items.id"), index=True)
    # Video items: which sampled frame this box came from.
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    # Normalized 0..1 [x1, y1, x2, y2] — resolution-independent, so boxes
    # survive thumbnailing and re-encoding.
    bbox: Mapped[str] = mapped_column(Text)
    class_name: Mapped[str] = mapped_column(String, index=True)
    # Optional refinement of class_name into an operator-defined class
    # (e.g. class_name="car", custom_class="delivery-van").
    custom_class: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id"), nullable=True, index=True
    )
    # Name proposed by an importer before a human confirms it (Takeout
    # people tags). Kept separate from identity_id so unverified guesses
    # never leak into the identity store.
    proposed_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # How the proposal was made: "unambiguous" (one face, one tag),
    # "clustered", "single-candidate", or None for plain detections.
    proposal_basis: Mapped[str | None] = mapped_column(String, nullable=True)
    # "auto" (detector), "human" (drawn/corrected), "import" (sidecar)
    source: Mapped[str] = mapped_column(String, default="auto", index=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # Explicitly rejected by a reviewer — kept as a negative example
    # rather than deleted, so the same bad proposal isn't re-suggested.
    rejected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # This annotation's embedding has been added to the identity's vector
    # collection — the idempotency marker for enrollment sweeps.
    enrolled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    item: Mapped[LibraryItem] = relationship(back_populates="annotations")
    identity: Mapped[Identity | None] = relationship()


class CustomClass(Base):
    """An operator-defined refinement of a detection class.

    Classification is k-NN over the same appearance embeddings the
    identity layer already computes — a custom class is just a labeled
    set of example crops, so defining one requires no training run.
    """

    __tablename__ = "custom_classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    # Detection class this refines ("car"); empty means it applies to any.
    parent_class: Mapped[str] = mapped_column(String, default="", index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # Minimum k-NN vote similarity to assign this class.
    threshold: Mapped[float] = mapped_column(Float, default=0.85)
    example_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class OperationRun(Base):
    """A long-running operator task: archive import, library indexing.

    Heartbeated to the database every batch, which is what makes these
    jobs observable rather than opaque: progress can be read from the web
    UI or a second terminal while the work happens in a third, and a run
    that died leaves its last known position behind instead of vanishing.
    """

    __tablename__ = "operation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    target: Mapped[str] = mapped_column(String, default="")
    phase: Mapped[str] = mapped_column(String, default="")
    # running | complete | interrupted | failed
    status: Mapped[str] = mapped_column(String, default="running", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    # Domain counters (faces detected, certain, matched…) as JSON.
    counters: Mapped[str] = mapped_column(Text, default="{}")
    # Per-phase elapsed seconds, so slow stages are identifiable after
    # the fact rather than guessed at.
    phase_timings: Mapped[str] = mapped_column(Text, default="{}")
    message: Mapped[str] = mapped_column(Text, default="")
    # Exact command to continue an interrupted run.
    resume_command: Mapped[str] = mapped_column(Text, default="")
    # Who was doing the work: enough to check liveness (same host) and,
    # later, to tell one site's runs from another's.
    pid: Mapped[int] = mapped_column(Integer, default=0)
    host: Mapped[str] = mapped_column(String, default="")

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100.0) if self.total else 0.0

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or self.updated_at
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def rate(self) -> float:
        """Items per second over the run so far."""
        return self.current / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float | None:
        if self.status != "running" or not self.total or self.rate <= 0:
            return None
        if self.is_stale:
            return None  # nobody is working on it; an ETA would be a lie
        return max(0.0, (self.total - self.current) / self.rate)

    @property
    def is_stale(self) -> bool:
        """A 'running' row that nothing is working on any more.

        Two signals. On the host that recorded the run, the pid answers
        immediately — waiting out a heartbeat timeout to notice a process
        that is provably gone helps nobody. Everywhere else (and when a
        recycled pid makes a dead run look alive) the cold heartbeat is
        the backstop.
        """
        if self.status != "running":
            return False
        from datetime import timezone as _tz

        from siteloom.health import hostname, process_alive

        if self.pid and self.host and self.host == hostname():
            if not process_alive(self.pid):
                return True

        now = datetime.now(_tz.utc).replace(tzinfo=None)
        return (now - self.updated_at).total_seconds() > 120


class TrainingRun(Base):
    """A completed training/evaluation run, for provenance in the UI."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String, index=True)  # "face-embed"|"face-detect"
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    identity_count: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
