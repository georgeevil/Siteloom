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

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    not_,
    or_,
    select,
)
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
    # Running sum of detection confidences; divided by detection_count it
    # gives the mean, which separates sustained evidence from one lucky
    # frame (CLD-40). Stored as a sum, not a mean, so per-frame increments
    # and event merges stay exact. 0 on rows predating the column —
    # `siteloom events retag` backfills them from Detection rows.
    confidence_sum: Mapped[float] = mapped_column(Float, default=0.0)
    # True if the event falls inside a known guest arrival window
    # (booking correlation, PRD §6.7) — used to suppress false alarms.
    guest_window: Mapped[bool] = mapped_column(Boolean, default=False)
    # Operator-recorded miss (CLD-16): an identifiable subject was present
    # but the system claimed no identity for it. Kept as a flag on the
    # event so accuracy queries stay one-table simple.
    missed_identity: Mapped[bool] = mapped_column(Boolean, default=False)
    missed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Operator sign-off (CLD-20): "I have looked at this, it needs nothing
    # further". Explicit rather than inferred from identity verdicts,
    # because an event with no identity links at all — an unmatched car —
    # has no verdicts to infer from and would otherwise sit in the queue
    # forever. Clearing is reversible; reopening nulls it.
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Significance gate (event noise reduction): ingest creates events as
    # insignificant ("ephemeral") and flips this once the event accumulates
    # enough detections/confidence/duration (EventConfig); the flip is
    # monotonic. The default triage view hides ephemeral events; nothing is
    # deleted. The column default is True so rows predating the column —
    # and writers that don't gate (Frigate consumer, tests) — stay visible;
    # `siteloom events retag` recomputes historical rows from stored counts.
    significant: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    camera: Mapped[Camera] = relationship(back_populates="events")
    detections: Mapped[list["Detection"]] = relationship(back_populates="event")
    identities: Mapped[list["EventIdentity"]] = relationship(back_populates="event")

    @property
    def active_identities(self) -> list["EventIdentity"]:
        """The identity claims that still stand on this event.

        Unlinked rows (CLD-36) keep their identity_id as the record of
        what was wrongly claimed, so anything asking "who is this event"
        — the triage list, search, review status — has to ask for the
        live claims rather than iterating the relationship. Misses (null
        identity_id) are not claims either.
        """
        return [
            link
            for link in self.identities
            if link.identity_id is not None and link.unlinked_at is None
        ]

    @property
    def mean_confidence(self) -> float | None:
        """Average detection confidence across the event, or None when it
        cannot be computed (no detections, or a row written before
        confidence_sum existed and not yet retagged)."""
        if self.detection_count <= 0 or self.confidence_sum <= 0:
            return None
        return self.confidence_sum / self.detection_count

    @property
    def review_status(self) -> str:
        """Where this event sits in the operator's review queue.

        Sign-off wins over everything: once an operator clears an event it
        leaves the queue even if it carries a wrong verdict, because the
        queue answers "what still needs me", not "what went wrong". The
        verdicts themselves are never overwritten, so accuracy reporting
        still sees them.

        Confirming every identity claim deliberately does *not* auto-clear
        — an operator can agree the system named someone correctly and
        still want the event escalated.

        `status_clause` below expresses the same rules in SQL for
        filtering and paging. The two must agree — tests compare them
        over the same rows.
        """
        if self.reviewed_at is not None:
            return "cleared"
        # An unlinked claim has already been corrected — its "wrong"
        # verdict is history, not outstanding work, so it must not pin
        # the event to `flagged` after the operator fixed it. Misses stay
        # in scope; `missed_identity` below covers them.
        verdicts = [
            link.verdict for link in self.identities if link.unlinked_at is None
        ]
        if self.missed_identity or "wrong" in verdicts:
            return "flagged"
        if any(v is not None for v in verdicts):
            return "reviewing"
        return "new"


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
    """One identity claim (or recorded miss) on an event.

    Two shapes share this table, distinguished by `identity_id`:

    * A match: `identity_id` set, `verdict` None/"confirmed"/"wrong".
    * A recorded miss (CLD-16's null-identity verdict rows): `identity_id`
      NULL, `verdict` = "missed", `identifier_key` naming which algorithm
      should have claimed something. Kept here rather than as a bare flag
      on Event so a miss is attributable — per-identifier recall (CLD-17)
      is not computable from "something was missed".

    A third state cuts across the first: `unlinked_at` set means the
    operator repudiated the claim (CLD-36). The row keeps its
    identity_id, similarity and matched_by — that is precisely the
    evidence of what the system got wrong, and the negatives-are-data
    rule applies here as much as it does to annotations — but it is no
    longer a claim, so `Event.active_identities` and the SQL clauses
    below skip it. Nulling identity_id instead would have collided with
    the miss shape above: a miss is "nothing was claimed", which is a
    different fact from "this was claimed and it was wrong".

    `Event.missed_identity` is maintained as a denormalized mirror of
    "this event has missed rows" so one-table accuracy queries and the
    triage status SQL stay simple; `set_missed` in the web layer is the
    single writer keeping them in step.
    """

    __tablename__ = "event_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id"), nullable=True, index=True
    )
    # Which identification algorithm produced (or should have produced)
    # this row. Set at ingest for matches, by the operator for misses.
    identifier_key: Mapped[str | None] = mapped_column(String, nullable=True)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    hit_count: Mapped[int] = mapped_column(Integer, default=1)
    # How the match was made: "plate" (OCR, wins outright per PRD §6.4) or
    # "visual" (cosine similarity). The resolver knows this at match time
    # and it cannot be reconstructed afterwards, so it is persisted here —
    # CLD-17's plate-vs-visual split reads straight off this column.
    matched_by: Mapped[str | None] = mapped_column(String, nullable=True)
    # True when this match is the one that taught the identity its plate
    # ("visual match learns its plate later"). If the operator then marks
    # this claim wrong, the learned plate is reverted — see the verdict
    # endpoint — because a mis-learned plate poisons every future
    # plate-first match for that number.
    learned_plate: Mapped[bool] = mapped_column(Boolean, default=False)
    # Human review of this identity claim (CLD-16): None = unreviewed,
    # "confirmed" or "wrong" ("missed" on null-identity rows). A wrong
    # verdict is persisted, never deleted (the Annotation provenance
    # philosophy — negatives are data), and it must not mutate the vector
    # store here; resolver-side learning from verdicts is separate work.
    verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    verdict_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # When the operator detached this claim (CLD-36). Set by unlink and by
    # reassign, which detaches the old claim and attaches a new one.
    unlinked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    event: Mapped[Event] = relationship(back_populates="identities")
    identity: Mapped[Identity | None] = relationship(back_populates="events")

    @property
    def is_miss(self) -> bool:
        return self.identity_id is None

    @property
    def is_active(self) -> bool:
        """A standing claim: an identity, not repudiated."""
        return self.identity_id is not None and self.unlinked_at is None


#: Review states an Event can be in, in triage order (worst first).
REVIEW_STATUSES = ("flagged", "reviewing", "new", "cleared")


def status_clause(status: str):
    """SQL form of `Event.review_status`, for filtering with correct paging.

    Filtering in Python after the query would page wrongly — the offset
    would count rows the operator never sees — so the triage filters have
    to be expressible in SQL. Keep in step with the property above.
    """
    # Unlinked claims are excluded here too — see `review_status`: a
    # corrected claim is not outstanding work.
    links = select(EventIdentity.id).where(
        EventIdentity.event_id == Event.id, EventIdentity.unlinked_at.is_(None)
    )
    has_wrong = links.where(EventIdentity.verdict == "wrong").exists()
    has_judged = links.where(EventIdentity.verdict.is_not(None)).exists()
    signed_off = Event.reviewed_at.is_not(None)
    open_ = Event.reviewed_at.is_(None)
    flagged = or_(Event.missed_identity.is_(True), has_wrong)

    if status == "cleared":
        return signed_off
    if status == "flagged":
        return and_(open_, flagged)
    if status == "reviewing":
        return and_(open_, not_(flagged), has_judged)
    if status == "new":
        return and_(open_, not_(flagged), not_(has_judged))
    raise ValueError(f"unknown review status: {status!r}")


def significance_clause():
    """Events past the significance gate — the default triage view.

    An orthogonal axis to `review_status` (like `unmatched_clause`): an
    ephemeral event still has a review status, it just isn't shown until
    the operator asks for ephemeral events.
    """
    return Event.significant.is_(True)


def unmatched_clause():
    """Events the identity layer never linked to anyone.

    Only real matches count — a recorded miss (null-identity row) is an
    operator saying nothing was matched, so it must not make the event
    look matched. Nor must an unlinked claim (CLD-36): an operator who
    detached the only match has said this event is unmatched, and the
    triage chip that surfaces such events is where it belongs.
    """
    return not_(
        select(EventIdentity.id)
        .where(
            EventIdentity.event_id == Event.id,
            EventIdentity.identity_id.is_not(None),
            EventIdentity.unlinked_at.is_(None),
        )
        .exists()
    )


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


class BackfillClip(Base):
    """One NVR recording window queued for backfill (PRD §6.6).

    Same two-phase shape as LibraryItem: a cheap scan registers windows
    as "pending" (keyed by the NVR's own event id, so re-scanning the
    same range adds nothing), and an expensive process phase downloads
    and ingests them in chronological order. That key is also what makes
    the operation resumable — an interrupted run's remaining clips are
    simply the ones still pending.
    """

    __tablename__ = "backfill_clips"
    __table_args__ = (UniqueConstraint("camera_id", "external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Siteloom camera id (config), not the NVR's internal camera id.
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    # NVR event id, or a synthetic "chunk:<iso>" id for full-range sweeps.
    external_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String, default="motion")
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime)
    # pending -> done | failed  (failed is terminal until retried, and
    # `attempts` distinguishes always-fails from not-retried-yet —
    # library indexer convention.)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


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
    # Processing attempts, so a file that fails every time is
    # distinguishable from one that has not been tried since it failed.
    attempts: Mapped[int] = mapped_column(Integer, default=0)
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


class User(Base):
    """An operator account (CLD auth milestone).

    Roles are a strict ladder — view < edit < admin — because the PoC
    needs "who may look, who may judge, who may reconfigure", not a
    permission matrix. Authentication is enabled by the existence of any
    User row: an empty table means the single-operator open mode the PoC
    started with, so nothing breaks before the first `siteloom users add`.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    # "scrypt$<salt-hex>$<hash-hex>" — see web/auth.py; never a raw secret.
    password_hash: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, default="view")  # view|edit|admin
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class WebSession(Base):
    """A logged-in browser session; the cookie holds only the token."""

    __tablename__ = "web_sessions"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    user: Mapped[User] = relationship()


class AuditLog(Base):
    """Who changed what, when — one row per mutating web request.

    Written by middleware rather than per-endpoint calls so a new POST
    route cannot forget to audit. `username` is denormalized on purpose:
    the row must still say who acted after the account is renamed or
    deleted. "(open)" records mutations made before auth was enabled.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    username: Mapped[str] = mapped_column(String, index=True)
    method: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String, index=True)
    status_code: Mapped[int] = mapped_column(Integer, default=0)


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
