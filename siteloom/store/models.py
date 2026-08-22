"""Event store models (PRD §6.7, relational half).

An Event is one tracked object's visit: the same track ID on the same
camera, spanning first_seen..last_seen, with N individual Detections.

Identity rows are the relational half of the identity store: labels,
plates, appearance stats. The embeddings themselves live in the vector
store (Qdrant local mode, see siteloom/identity/vectors.py); the two are
joined by Identity.id, which is used as the vector point id's payload.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    not_,
    or_,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Naive UTC, the shape every timestamp column in here stores.

    The contract (CLD-100): **naive UTC everywhere in the store; convert
    at the display boundary via the site zone (`siteloom/localtime.py`,
    the `local_time` Jinja global); convert at input boundaries (NVR,
    iCal, operator forms) on the way in.** No column carries an offset,
    no comparison mixes frames, and no code below the web layer touches
    the site timezone.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    # Operator-recorded tracking failure (CLD-103): this event's crops are
    # not all the same subject. Distinct from `missed_identity`, which says
    # a subject went unnamed — here the event itself is wrong, and no
    # amount of identity work on it can be right.
    #
    # It is the corpus contribution the tuning harness needs. An operator
    # marking this is already looking at the event; the mark carries the
    # camera and time window, which is exactly a `track_ab.py` clip
    # definition. `track_ab.py suggest` reads these rows.
    multi_subject: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True
    )
    multi_subject_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    # Operator sign-off (CLD-20): "I have looked at this, it needs nothing
    # further". Explicit rather than inferred from identity verdicts,
    # because an event with no identity links at all — an unmatched car —
    # has no verdicts to infer from and would otherwise sit in the queue
    # forever. Clearing is reversible; reopening nulls it.
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # The incident this event was escalated into (CLD-96), or NULL — which
    # is what almost every event is. Escalation is deliberately NOT a fifth
    # `review_status`: one incident links many events across cameras, and a
    # per-event status could not express that (see `Incident`). So the
    # status property and its SQL twin below are untouched by this column.
    incident_id: Mapped[int | None] = mapped_column(
        ForeignKey("incidents.id"), nullable=True, index=True
    )
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
    incident: Mapped["Incident | None"] = relationship(back_populates="events")

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
    # Vehicle fingerprint color read (CLD-254), per frame like plate
    # reads: the visit-level answer is display-time grouping, never a
    # column. All NULL together means "not measured" — fingerprinting
    # off, a non-vehicle class, or a pre-column row — which every reader
    # must render as nothing, not as unknown. A NULL `color_name` with a
    # `color_reason` is a real read that named no color (grayscale/IR
    # crop, too-small crop). The measurements (`color_chroma`,
    # `color_crop_px`, `color_saturation`) and the floors they were
    # judged against (`color_min_px`, `color_chroma_floor`) are kept on
    # every read, refused ones included — moving a floor is a question
    # about existing rows, never a re-run (the PlateRead discipline).
    color_name: Mapped[str | None] = mapped_column(String, nullable=True)
    color_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    color_chroma: Mapped[float | None] = mapped_column(Float, nullable=True)
    color_saturation: Mapped[float | None] = mapped_column(Float, nullable=True)
    color_crop_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    color_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    color_min_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    color_chroma_floor: Mapped[float | None] = mapped_column(Float, nullable=True)

    event: Mapped[Event] = relationship(back_populates="detections")


#: Values `Identity.plate_source` takes. NULL is a fourth, real state —
#: a plate recorded before provenance was tracked — and every reader has
#: to render it as unknown rather than as any of these.
PLATE_SOURCE_MINT = "mint"  # read when the identity was first seen
PLATE_SOURCE_LEARNED = "learned"  # a later visual match learned it
PLATE_SOURCE_OPERATOR = "operator"  # set, or deliberately cleared, by a person
PLATE_SOURCES = (PLATE_SOURCE_MINT, PLATE_SOURCE_LEARNED, PLATE_SOURCE_OPERATOR)


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
    # Where `plate` came from (CLD-134). NULL on rows written before this
    # was tracked — unknown provenance, not "never set", the same answer
    # CLD-84 gave for vectors that predate their marker. "operator" is
    # load-bearing rather than informational: it is what stops the
    # resolver re-learning a plate an operator just cleared, since an
    # empty plate is precisely the condition the learn path fires on.
    plate_source: Mapped[str | None] = mapped_column(String, nullable=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    appearance_count: Mapped[int] = mapped_column(Integer, default=0)
    # How many embeddings this identity has in the vector store (capped).
    vector_count: Mapped[int] = mapped_column(Integer, default=0)
    # The representative crop. Despite sharing a name with
    # `Event.best_crop_path`, the two have never meant the same thing:
    # the event field is best-confidence-wins, maintained per frame
    # (ingest.py), while this one was first-write-wins at every writer —
    # which is how an identity ended up wearing the face of the wrong
    # match that founded it (CLD-137). It is now "the current cover":
    # re-derived when the event that supplied it stops being this
    # identity's, or chosen outright by an operator.
    best_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # An operator picked the cover above, so recompute leaves it alone.
    # One bit rather than a `cover_source` triple: "auto-picked at mint"
    # and "auto-picked by recompute" are a distinction no screen makes
    # and no operator can act on (contrast `plate_source`, where all
    # three values change what you do about a wrong plate — CLD-134).
    cover_locked: Mapped[bool] = mapped_column(Boolean, default=False)

    events: Mapped[list["EventIdentity"]] = relationship(back_populates="identity")

    @property
    def display_name(self) -> str:
        return self.label or f"unknown-{self.identifier_key}-{self.id}"


#: Name of the partial unique index on EventIdentity. Shared with
#: store/db.py, which must create it *after* the duplicate-claim repair
#: pass and must not create it during the table rebuild.
ACTIVE_CLAIM_INDEX = "uq_event_identities_active_claim"


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
    different fact from "this was claimed and it was wrong". Only one
    such standing claim per (event, identity) may exist, and
    `ACTIVE_CLAIM_INDEX` below enforces it — partially, over
    `unlinked_at IS NULL`, because the other two shapes are legitimately
    repeatable (CLD-133).

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

    __table_args__ = (
        # One standing claim per (event, identity) — the constraint that
        # makes CLD-133's stacked duplicates unrepresentable rather than
        # merely discouraged. Partial, because the other two shapes in
        # this table are legitimately repeatable: an operator may
        # repudiate the same pairing more than once, and those unlinked
        # rows are evidence, not garbage (CLD-36); and NULL identity_id
        # miss rows are distinct under a unique index by SQL's NULL
        # semantics, so several identifiers can each record a miss on one
        # event. Both dialect kwargs are set: SQLite runs it today,
        # Postgres spells the same partial index differently.
        Index(
            ACTIVE_CLAIM_INDEX,
            "event_id",
            "identity_id",
            unique=True,
            sqlite_where=text("unlinked_at IS NULL"),
            postgresql_where=text("unlinked_at IS NULL"),
        ),
    )

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


#: An incident's lifecycle. Two states, because the question it answers is
#: "is anyone still working this" — anything finer is a workflow tool, and
#: a property manager with one console does not have a workflow.
INCIDENT_STATUSES = ("open", "closed")


class Incident(Base):
    """Something happened here — kept, so it can be found again (CLD-96).

    The Escalate action on the triage rail had no semantics for a year
    because nobody could answer "escalate to whom". For a property manager
    the honest answer is: nobody, immediately — and somebody, later. An
    insurer, the police, a tenant dispute six weeks on. That rules out a
    notification (a broker nobody is subscribed to) and a review queue
    (state that means nothing once the shift ends), and leaves a record.

    Why this is its own table rather than a fifth `Event.review_status`:

    * **One incident links many events.** A vehicle arriving, someone at
      the door, and a noise episode forty minutes later are one incident
      across three cameras. A per-event flag cannot say that, and saying
      it is the entire reason this exists. `Event.incident_id` is the
      link; the status property and `status_clause` are left alone.
    * **It has a lifecycle of its own.** `Event.reviewed_at` says someone
      looked; an incident's closure says how it ended.
    * **It carries prose** (`IncidentNote`). Every other field here is
      reconstructible from the events; the notes are the part worth
      anything to a reader who was not there.

    `opened_by`/`closed_by` are denormalized usernames for the same reason
    `AuditLog.username` is: the record must still say who judged this
    after the account is renamed or deleted. "(open)" records a judgement
    made before auth was enabled.

    Deleting an incident must not delete its events — the events are the
    record, the incident is an interpretation of them — so `events` carries
    no delete cascade and the member rows are detached instead. `notes`
    *are* part of the interpretation and go with it.
    """

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="open", index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    opened_by: Mapped[str] = mapped_column(String, default="")
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_by: Mapped[str | None] = mapped_column(String, nullable=True)

    events: Mapped[list["Event"]] = relationship(back_populates="incident")
    notes: Mapped[list["IncidentNote"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        order_by="IncidentNote.at",
    )

    @property
    def is_open(self) -> bool:
        return self.status == "open"


class IncidentNote(Base):
    """One entry of an incident's prose, appended.

    Append-only rather than one editable blob: the notes are the evidence
    half of the record, and a record that can be silently rewritten is
    worth less to the reader it was written for. `kind` marks the two
    entries the console writes itself — the closing note and the reason a
    closed incident was reopened — so an export can render the lifecycle
    and the commentary as one chronology instead of two.
    """

    __tablename__ = "incident_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    author: Mapped[str] = mapped_column(String, default="")
    # "note" | "close" | "reopen"
    kind: Mapped[str] = mapped_column(String, default="note")
    body: Mapped[str] = mapped_column(Text, default="")

    incident: Mapped[Incident] = relationship(back_populates="notes")


#: Verdicts an operator can record on a plate read. Same vocabulary as
#: `EventIdentity.verdict` deliberately — "was this claim right" is the
#: same question, and two spellings of it would need two readers.
PLATE_VERDICTS = ("confirmed", "wrong")


class PlateRead(Base):
    """One plate-OCR attempt on one detection crop, kept whether or not
    it produced a plate (CLD-85).

    Before this table the OCR threw away everything by which it could be
    judged: the detector's box confidence picked a region and was
    discarded, no OCR confidence was captured, `normalize_plate` is lossy
    and irreversible, and a read under the four-character floor returned
    None leaving no trace that a read had been attempted. What survived
    downstream was `Identity.plate` (one string, write-once),
    `EventIdentity.matched_by` and `learned_plate` — enough to say how
    many matches were plate matches, not enough to say whether any of
    them were right.

    Three things make this row worth writing:

    * **Failures are rows.** `reason` records why nothing came back
      (`no-box`, `empty-crop`, `no-text`, `too-short`) — the same
      negatives-are-data philosophy as `Annotation.rejected`. Motorcycle
      plates are the short/angled case that fails the floor, so the rows
      that answer CLD-9 are precisely the ones the old code dropped.
    * **`raw_text` is kept beside `text`.** Normalization strips every
      non-[A-Z0-9] character irreversibly, which makes a near-miss read
      indistinguishable from a clean one. `min_chars` rides along so a
      row says which bar it was judged against, and lowering that bar is
      a re-query rather than a re-run.
    * **`crop_path` is a third image.** Not `Detection.crop_path`, which
      is both the display thumbnail and the embedder input ("one crop,
      two jobs") — this is the plate sub-region, so a human can see what
      the OCR saw without disturbing a single stored vector.

    `class_name` and `camera_id` are denormalized off the detection so
    the review screen's class filter (isolate motorcycles) is one indexed
    predicate rather than a join through a mutable Event.class_name.
    """

    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    # The detection whose crop was read. Nullable because a read is
    # meaningful without one — and because a future replay path may hand
    # over a crop that is not a Detection row.
    detection_id: Mapped[int | None] = mapped_column(
        ForeignKey("detections.id"), nullable=True, index=True
    )
    camera_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    class_name: Mapped[str] = mapped_column(String, default="", index=True)
    # Which identifier's OCR this was ("vehicle"), so per-identifier
    # accuracy stays computable if a second plate identifier ever exists.
    identifier_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    # Exactly what the OCR returned, before normalization.
    raw_text: Mapped[str | None] = mapped_column(String, nullable=True)
    # normalize_plate(raw_text) — set even when it fell under the floor.
    text: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # True when this read was handed to the resolver as a plate.
    accepted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # One of plates.REASONS when nothing usable came back, else NULL.
    reason: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # Plate-region box confidence, and mean per-character OCR probability
    # where the installed OCR reports one. Both nullable: a confidence
    # that was never reported must read as absent, not as zero.
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Image-quality measurements, taken on every read whether or not any
    # floor is configured — because the floors cannot be chosen without
    # them. OCR confidence describes how sure the model was; these three
    # describe whether the picture could have carried the answer, which
    # is the distinction behind "confident and still wrong": a smeared
    # 60-pixel plate reports a high mean confidence about characters it
    # interpolated.
    #
    # `ocr_min_confidence` is the weakest character's probability from
    # the same array `ocr_confidence` averages — the number the mean
    # hides, and the one worth gating on.
    ocr_min_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Size of the plate region in source pixels, and variance of its
    # Laplacian (blur). Nullable for the same reason as the confidences,
    # and additionally because rows written before these columns existed
    # have no measurement — not a measurement of zero.
    plate_width: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    plate_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sharpness: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    # The floor in force when this read was judged.
    min_chars: Mapped[int] = mapped_column(Integer, default=4)
    # The plate sub-crop (see the class docstring) and the detection crop
    # it was cut from, so a reviewer can see the vehicle as well.
    crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    source_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Operator judgement of the read itself — the thing that turns CLD-9
    # from "eyeball ten crops once" into "judge 20 rows", persistently.
    verdict: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    verdict_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # What the plate actually says, typed by the operator (normalized with
    # the same `normalize_plate` the OCR's output went through, so the two
    # columns are comparable character by character — ground truth for
    # "which characters does the OCR confuse"). Like `verdict`, it changes
    # nothing in the identity store: `Identity.plate` is write-once, and
    # unwinding a plate match is a larger decision than judging a read.
    corrected_text: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    event: Mapped[Event] = relationship()

    @property
    def display_text(self) -> str:
        """What to show for a read, preferring the raw OCR output.

        A rejected read has no `text` by definition; showing the raw
        string is the whole point of keeping it, and "(no text)" is a
        different fact from an empty cell.
        """
        return self.raw_text or self.text or ""


class PlateWatch(Base):
    """A plate the operator wants flagged on sight.

    The row is intent, not evidence: sightings stay in `PlateRead` and
    are joined at read time, so the watchlist can never disagree with the
    reads table about when a plate was last seen. `plate` is stored
    normalized — the one form matching uses — and unique, so watching a
    plate twice is an update, not a second row.

    Matching happens where reads are persisted (`ingest.py`): the first
    accepted read of a watched plate on an event fires the
    `plate.watchlist` webhook and an MQTT message, once per event —
    a 30-second visit is one alarm, not sixty.
    """

    __tablename__ = "plate_watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plate: Mapped[str] = mapped_column(String, unique=True, index=True)
    # Why this plate is watched ("banned contractor van"). Free text for
    # a human; the alarm payload carries both so an automation can route
    # on the label without a second lookup.
    label: Mapped[str] = mapped_column(String, default="")
    note: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)


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
    """A guest booking (PRD §6.7 guest-correlation), from iCal or by hand."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String, unique=True, index=True)
    summary: Mapped[str] = mapped_column(String, default="")
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime, index=True)
    # Provenance (CLD-90): "ical" for rows the feed owns, "manual" for an
    # operator's correction. The iCal sync keys on `uid`, which is unique,
    # so without this column a feed carrying a colliding UID would silently
    # overwrite a hand-entered booking — and an operator's correction that
    # the next sync reverts is worse than no correction at all. Defaults to
    # "ical" because every row predating the column came from the feed.
    source: Mapped[str] = mapped_column(String, default="ical", index=True)

    @property
    def is_manual(self) -> bool:
        """Whether an operator owns this row, and may therefore edit it.

        Editing an iCal row in place is deliberately not offered: the feed
        is the source of truth for its own rows and the next sync would
        revert the edit. The fix for a wrong feed is a manual booking.
        """
        return self.source == "manual"


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


#: Who set `Annotation.verified`. `source` cannot answer this — it is the
#: provenance of the *box* ("auto"/"human"/"import") and no reviewer ever
#: rewrites it, so a human confirming an imported annotation stays
#: "import" forever (CLD-95).
#:
#: * "human"  — a person clicked confirm/classify in the review UI.
#: * "import" — an importer verified it with nobody looking (the Takeout
#:              pass-1 auto-verify: one face, one people-tag).
#: * "auto"   — reserved for a future in-product automatic verifier (a
#:              model confirming its own proposals). Nothing writes it
#:              yet; it exists so the next auto-verifying path does not
#:              reuse "import" and quietly re-merge the two.
VERIFIED_BY_HUMAN = "human"
VERIFIED_BY_IMPORT = "import"
VERIFIED_BY_AUTO = "auto"
VERIFIED_BY = (VERIFIED_BY_HUMAN, VERIFIED_BY_IMPORT, VERIFIED_BY_AUTO)


class Annotation(Base):
    """A box on a library item — machine-detected or human-drawn.

    One table serves detection review, identity labeling, custom-class
    labeling, and face-training data. `source` records provenance and
    `verified` records sign-off; training exports only ever read verified
    rows, which is what keeps auto-assignments from silently becoming
    ground truth.

    `verified_by`/`verified_at` record *who* signed off and when, because
    `verified` alone conflates a person clicking confirm with an importer
    auto-verifying itself — and `training/dataset.py` reads both as ground
    truth. The invariant to preserve: **`verified_by` is set exactly when
    `verified` is True**, which is what lets a query trust it (see
    `mark_verified`/`clear_verified`).
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
    # One of VERIFIED_BY, or NULL when the row is not verified. Nullable
    # on purpose: an unverified row has no verifier, and a pre-CLD-95
    # database has rows whose verifier was never recorded — NULL there
    # means "unknown", which a defaulted column could not say.
    verified_by: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Explicitly rejected by a reviewer — kept as a negative example
    # rather than deleted, so the same bad proposal isn't re-suggested.
    #
    # Rejection deliberately does NOT get its own provenance column.
    # `rejected` already carries it: no importer ever rejects (the Takeout
    # passes only ever write proposals), so a rejected row is a human act
    # by construction. Adding `rejected_by` would record the same fact
    # twice and cost the invariant above — `verified_by` set exactly when
    # `verified` is True is checkable; "set when verified or rejected" is
    # not. Rejecting therefore *clears* verified_by, like any other return
    # to unverified.
    rejected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # This annotation's embedding has been added to the identity's vector
    # collection — the idempotency marker for enrollment sweeps.
    enrolled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    item: Mapped[LibraryItem] = relationship(back_populates="annotations")
    identity: Mapped[Identity | None] = relationship()

    def mark_verified(self, by: str, at: datetime | None = None) -> None:
        """Sign this row off, recording who did it and when.

        The only supported way to set `verified` — going through it is
        what keeps `verified_by` from drifting out of step with
        `verified`. `by` is re-stamped every time, so a human confirming
        an annotation the importer had already auto-verified ends up
        "human": that is precisely the transition the old schema lost.
        """
        if by not in VERIFIED_BY:
            raise ValueError(f"unknown verifier {by!r}; expected one of {VERIFIED_BY}")
        self.verified = True
        self.verified_by = by
        self.verified_at = at or _utcnow()

    def clear_verified(self) -> None:
        """Return the row to unverified, dropping the sign-off with it.

        Used by reject and by un-review. Leaving a stale verifier on a row
        whose `verified` is False would make the pair unreadable.
        """
        self.verified = False
        self.verified_by = None
        self.verified_at = None


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
    # Which process, not just which pid (CLD-57): the OS-reported start
    # time of the recording process, as `health.process_identity` renders
    # it. A pid outlives its process and the OS hands it out again, so
    # without this a cancel can signal a stranger. Empty means the row
    # cannot prove whose pid that is — every row written before this
    # column existed (the migration fills them with ''), and any row
    # written where the platform would not report a start time.
    process_start: Mapped[str] = mapped_column(String, default="")

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

        Two signals. On the host that recorded the run, the recorded
        process answers immediately — waiting out a heartbeat timeout to
        notice a process that is provably gone helps nobody, and a pid
        now worn by an unrelated process is just as provably gone
        (`process_verdict`, the same question `request_cancel` asks).
        Everywhere else, and whenever the identity cannot be established
        at all, the cold heartbeat is the backstop.
        """
        if self.status != "running":
            return False
        from datetime import timezone as _tz

        from siteloom.health import PROVEN_GONE, hostname, process_verdict

        if self.pid and self.host and self.host == hostname():
            if process_verdict(self.pid, self.process_start or "") in PROVEN_GONE:
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
