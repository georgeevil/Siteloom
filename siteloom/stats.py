"""Accuracy readout over identity claims (CLD-17).

Turns a soak's data into an answer to "is it holding up?". Every column
this reads already exists — `EventIdentity.matched_by`, `.learned_plate`,
`.verdict`, `.identifier_key`, and the null-`identity_id` rows that record
a miss — they were simply never read in aggregate.

Pure query functions returning plain dataclasses, deliberately outside the
web layer: the arithmetic is the part worth testing, and a `siteloom stats`
CLI would want exactly the same numbers.

Three rules are baked into the shapes here rather than left to callers:

1. **A rate never travels without its denominator.** Operators review
   suspicious matches more than obvious ones, so "precision over reviewed
   links" is a biased sample. `wrong_rate` is None when nothing was
   reviewed, and `reviewed`/`claims` ride alongside it so a renderer
   cannot print a bare percentage without the coverage that qualifies it.
2. **Similarity distributions are never pooled across identifiers.** Face
   lives near 0.36 and generic near 0.8+; one histogram over both is a
   picture of the configuration, not of the data.
3. **Plate matches are excluded from the similarity view.** A plate match
   carries a synthetic similarity (the resolver mints one so the column is
   never null), and mixing those into the histogram would put a spike at a
   made-up value right where the operator is reading the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import and_, case, func, select

from siteloom.store import Camera, Event, EventIdentity

#: Width of a similarity histogram bin over the 0..1 cosine range.
BIN_WIDTH = 0.05


def _count_if(condition) -> object:
    """SUM(CASE WHEN ...) — COUNT(CASE ...) counts zeros on some backends."""
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _window(stmt, since: datetime | None, until: datetime | None):
    if since is not None:
        stmt = stmt.where(Event.first_seen >= since)
    if until is not None:
        stmt = stmt.where(Event.first_seen < until)
    return stmt


@dataclass
class IdentifierStat:
    """One identifier's day: what it claimed, and what that was worth."""

    key: str
    claims: int = 0  # links carrying an identity
    confirmed: int = 0
    wrong: int = 0
    misses: int = 0  # operator-recorded, null-identity rows
    plate_matches: int = 0
    visual_matches: int = 0
    learned_plates: int = 0

    @property
    def reviewed(self) -> int:
        return self.confirmed + self.wrong

    @property
    def unreviewed(self) -> int:
        return max(0, self.claims - self.reviewed)

    @property
    def unattributed(self) -> int:
        """Claims with no `matched_by` — rows written before the column."""
        return max(0, self.claims - self.plate_matches - self.visual_matches)

    @property
    def wrong_rate(self) -> float | None:
        """Share of *reviewed* claims judged wrong, or None if none were.

        None rather than 0.0: an unreviewed identifier has not been shown
        to be perfect, and a zero here reads exactly like that.
        """
        return (self.wrong / self.reviewed) if self.reviewed else None

    @property
    def coverage(self) -> float | None:
        """Share of claims an operator actually judged — the denominator
        that qualifies `wrong_rate`."""
        return (self.reviewed / self.claims) if self.claims else None

    @property
    def recall_denominator(self) -> int:
        """Claims plus recorded misses: what the identifier could have got."""
        return self.claims + self.misses


def identifier_stats(
    session, since: datetime | None = None, until: datetime | None = None
) -> list[IdentifierStat]:
    """Per-identifier claim/verdict/miss counts over a window."""
    ei = EventIdentity
    has_identity = ei.identity_id.is_not(None)
    stmt = (
        select(
            ei.identifier_key,
            _count_if(has_identity).label("claims"),
            _count_if(ei.verdict == "confirmed").label("confirmed"),
            _count_if(and_(ei.verdict == "wrong", has_identity)).label("wrong"),
            _count_if(ei.identity_id.is_(None)).label("misses"),
            _count_if(and_(has_identity, ei.matched_by == "plate")).label("plate"),
            _count_if(and_(has_identity, ei.matched_by == "visual")).label("visual"),
            _count_if(ei.learned_plate.is_(True)).label("learned"),
        )
        .join(Event, Event.id == ei.event_id)
        .group_by(ei.identifier_key)
        .order_by(ei.identifier_key)
    )
    rows = session.execute(_window(stmt, since, until)).all()
    return [
        IdentifierStat(
            key=key or "unattributed",
            claims=claims,
            confirmed=confirmed,
            wrong=wrong,
            misses=misses,
            plate_matches=plate,
            visual_matches=visual,
            learned_plates=learned,
        )
        for key, claims, confirmed, wrong, misses, plate, visual, learned in rows
    ]


@dataclass
class CameraStat:
    camera_id: str
    name: str
    events: int = 0
    matched_events: int = 0
    misses: int = 0

    @property
    def unmatched_events(self) -> int:
        return max(0, self.events - self.matched_events)


def camera_stats(
    session, since: datetime | None = None, until: datetime | None = None
) -> list[CameraStat]:
    """Per-camera event volume and how much of it got an identity.

    Counted over events, not links: "how many visits did we put a name to"
    is the operator's question, and an event with three links to the same
    identity is still one visit.
    """
    matched = (
        select(EventIdentity.id)
        .where(
            EventIdentity.event_id == Event.id,
            EventIdentity.identity_id.is_not(None),
        )
        .exists()
    )
    missed = (
        select(EventIdentity.id)
        .where(
            EventIdentity.event_id == Event.id,
            EventIdentity.identity_id.is_(None),
        )
        .exists()
    )
    stmt = (
        select(
            Event.camera_id,
            func.coalesce(Camera.name, Event.camera_id),
            func.count(Event.id),
            _count_if(matched),
            _count_if(missed),
        )
        .join(Camera, Camera.id == Event.camera_id, isouter=True)
        .group_by(Event.camera_id, Camera.name)
        .order_by(Event.camera_id)
    )
    return [
        CameraStat(
            camera_id=camera_id, name=name, events=events, matched_events=m, misses=miss
        )
        for camera_id, name, events, m, miss in session.execute(
            _window(stmt, since, until)
        ).all()
    ]


@dataclass
class ReviewCoverage:
    """How much of the window an operator actually worked through.

    `cleared_without_verdict` is a first-class number here, not an
    oversight. Clearing an event needs no verdict — deliberately, since
    most events have nothing to judge — which means a soak can be worked
    to zero while producing no accuracy data at all. Keeping clearing
    cheap and making that gap visible beats forcing a verdict nobody has.
    """

    events: int = 0
    cleared: int = 0
    cleared_without_verdict: int = 0
    flagged: int = 0
    total_links: int = 0
    reviewed_links: int = 0

    @property
    def open(self) -> int:
        return max(0, self.events - self.cleared)

    @property
    def link_coverage(self) -> float | None:
        return (self.reviewed_links / self.total_links) if self.total_links else None


def review_coverage(
    session, since: datetime | None = None, until: datetime | None = None
) -> ReviewCoverage:
    has_verdict = (
        select(EventIdentity.id)
        .where(
            EventIdentity.event_id == Event.id,
            EventIdentity.verdict.is_not(None),
        )
        .exists()
    )
    events_stmt = select(
        func.count(Event.id),
        _count_if(Event.reviewed_at.is_not(None)),
        _count_if(and_(Event.reviewed_at.is_not(None), ~has_verdict)),
        _count_if(Event.missed_identity.is_(True)),
    ).select_from(Event)
    events, cleared, cleared_bare, flagged = session.execute(
        _window(events_stmt, since, until)
    ).one()

    links_stmt = (
        select(
            func.count(EventIdentity.id),
            _count_if(EventIdentity.verdict.is_not(None)),
        )
        .select_from(EventIdentity)
        .join(Event, Event.id == EventIdentity.event_id)
    )
    total_links, reviewed_links = session.execute(
        _window(links_stmt, since, until)
    ).one()

    return ReviewCoverage(
        events=events,
        cleared=cleared,
        cleared_without_verdict=cleared_bare,
        flagged=flagged,
        total_links=total_links,
        reviewed_links=reviewed_links,
    )


@dataclass
class Bin:
    low: float
    high: float
    count: int = 0
    confirmed: int = 0
    wrong: int = 0

    @property
    def label(self) -> str:
        return f"{self.low:.2f}"


@dataclass
class Histogram:
    """Visual-match similarity distribution for one identifier.

    `wrong_above` and `confirmed_near` are computed from the raw
    similarities, not from the bins. A bin straddles the threshold far
    more often than it aligns to it — face sits at 0.36 against 0.05-wide
    bins — so deriving them from bin edges silently answers a question
    about 0.35 or 0.40 while claiming to answer one about 0.36.
    """

    key: str
    threshold: float | None
    bins: list[Bin] = field(default_factory=list)
    total: int = 0
    #: Wrong verdicts that still cleared the threshold — what raising the
    #: cutoff could remove.
    wrong_above: int = 0
    #: Confirmed matches within one bin-width above the threshold — what
    #: raising it would cost.
    confirmed_near: int = 0

    @property
    def peak(self) -> int:
        return max((b.count for b in self.bins), default=0)


def similarity_histograms(
    session,
    thresholds: dict[str, float] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Histogram]:
    """One histogram per identifier — never pooled, never including plates.

    `thresholds` maps identifier key to its configured cutoff so the view
    can mark where the line currently sits; an unknown key just renders
    without one.
    """
    thresholds = thresholds or {}
    ei = EventIdentity
    stmt = (
        select(ei.identifier_key, ei.similarity, ei.verdict)
        .join(Event, Event.id == ei.event_id)
        .where(ei.identity_id.is_not(None), ei.matched_by != "plate")
    )
    rows = session.execute(_window(stmt, since, until)).all()

    grouped: dict[str, list[tuple[float, str | None]]] = {}
    for key, similarity, verdict in rows:
        grouped.setdefault(key or "unattributed", []).append((similarity, verdict))

    bin_count = int(round(1.0 / BIN_WIDTH))
    histograms = []
    for key in sorted(grouped):
        threshold = thresholds.get(key)
        bins = [
            Bin(low=i * BIN_WIDTH, high=(i + 1) * BIN_WIDTH) for i in range(bin_count)
        ]
        wrong_above = confirmed_near = 0
        for similarity, verdict in grouped[key]:
            index = min(bin_count - 1, max(0, int(similarity / BIN_WIDTH)))
            bins[index].count += 1
            if verdict == "confirmed":
                bins[index].confirmed += 1
            elif verdict == "wrong":
                bins[index].wrong += 1
            if threshold is None:
                continue
            if verdict == "wrong" and similarity >= threshold:
                wrong_above += 1
            elif verdict == "confirmed" and threshold <= similarity < threshold + BIN_WIDTH:
                confirmed_near += 1
        histograms.append(
            Histogram(
                key=key,
                threshold=threshold,
                bins=bins,
                total=len(grouped[key]),
                wrong_above=wrong_above,
                confirmed_near=confirmed_near,
            )
        )
    return histograms
