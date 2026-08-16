"""One implementation of "these two rows are the same claim" (CLD-133).

The fold used to exist only inline in the ingest de-frag merge, so the
identity-merge endpoint re-pointed rows with a blind UPDATE and stacked
duplicates instead of collapsing them. Everything that can produce a
second active EventIdentity row for one (event, identity) pair goes
through here, and the partial unique index behind it (models.py) makes
that structural rather than a convention.

Imports models directly, never the `siteloom.store` package: the package
__init__ imports db.py, which imports this module.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from siteloom.store.models import EventIdentity

#: Verdict strength. A human verdict always outranks no verdict — folding
#: must never erase a review — and an affirmative identification outranks
#: a repudiation of the same pairing. Deliberately not CLD-133's literal
#: "confirmed > none > wrong": under that ordering, folding an unreviewed
#: keeper with a `wrong` loser yields None, which drops the event out of
#: the `flagged` triage bucket and discards the operator's work.
_VERDICT_RANK: dict[str | None, int] = {None: 0, "wrong": 1, "confirmed": 2}

#: Match-mode strength. A plate read outranks everything (PRD §6.4), and
#: a row whose first frame *minted* the identity (matched_by None) picks
#: up any later mode. "human" and "visual" are deliberately equal:
#: promoting one over the other would move rows between the two buckets
#: `siteloom/stats.py` reads straight off this column.
_MATCH_RANK: dict[str | None, int] = {None: 0, "visual": 1, "human": 1, "plate": 2}


def fold_claim(keeper: EventIdentity, loser: EventIdentity) -> None:
    """Combine two standing claims into `keeper`, in place.

    `loser` is left for the caller to delete: the three callers manage
    their own flush ordering around the delete, and doing it here would
    take that away from them.
    """
    if keeper is loser:
        raise ValueError("cannot fold a claim into itself")
    if keeper.event_id != loser.event_id:
        raise ValueError("cannot fold claims on different events")
    if keeper.unlinked_at is not None or loser.unlinked_at is not None:
        # An unlinked row is a closed record of a repudiated claim
        # (CLD-36), not a claim; folding one would resurrect what an
        # operator detached, or bury it inside a live row.
        raise ValueError("cannot fold an unlinked claim")
    keeper.hit_count += loser.hit_count
    keeper.similarity = max(keeper.similarity, loser.similarity)
    if _MATCH_RANK.get(loser.matched_by, 0) > _MATCH_RANK.get(keeper.matched_by, 0):
        keeper.matched_by = loser.matched_by
    keeper.learned_plate = keeper.learned_plate or loser.learned_plate
    if keeper.identifier_key is None:
        keeper.identifier_key = loser.identifier_key
    if _VERDICT_RANK.get(loser.verdict, 0) > _VERDICT_RANK.get(keeper.verdict, 0):
        # The verdict and its timestamp move together — a verdict_at
        # belonging to a different verdict says when a review that never
        # happened happened.
        keeper.verdict = loser.verdict
        keeper.verdict_at = loser.verdict_at


def record_sighting(
    link: EventIdentity,
    *,
    similarity: float,
    matched_by: str | None,
    learned_plate: bool = False,
) -> None:
    """Add one more observation of an already-standing claim."""
    link.hit_count += 1
    link.similarity = max(link.similarity, similarity)
    if _MATCH_RANK.get(matched_by, 0) > _MATCH_RANK.get(link.matched_by, 0):
        link.matched_by = matched_by
    link.learned_plate = link.learned_plate or learned_plate


def active_claim(
    session: Session, event_id: int, identity_id: int | None
) -> EventIdentity | None:
    """The standing claim for this pair, if the event carries one."""
    return (
        session.query(EventIdentity)
        .filter_by(event_id=event_id, identity_id=identity_id)
        .filter(EventIdentity.unlinked_at.is_(None))
        .first()
    )


def insert_claim(session: Session, row: EventIdentity) -> tuple[EventIdentity, bool]:
    """Add `row`, or fold it into the claim a racing writer just made.

    Returns (surviving_row, created). `created` is False when the insert
    lost the race, which is what keeps the once-per-pairing publish in
    ingest.py honest — the winner publishes, the loser does not.

    The savepoint is what makes the conflict recoverable: without it the
    failed statement dooms the caller's whole transaction, which for
    ingest spans a batch of unrelated rows. After the rollback `row` is
    expunged, so callers must use the returned object, never the one
    they passed in.
    """
    # Read the observation off the row before the flush: a failed INSERT
    # need not leave column defaults applied to the instance.
    similarity = row.similarity
    matched_by = row.matched_by
    learned_plate = bool(row.learned_plate)
    try:
        with session.begin_nested():
            session.add(row)
            # Flush inside the savepoint so the conflict surfaces here
            # rather than in some later autoflush.
            session.flush()
        return row, True
    except IntegrityError:
        winner = active_claim(session, row.event_id, row.identity_id)
        if winner is None:  # a different constraint failed — not ours
            raise
        record_sighting(
            winner,
            similarity=similarity,
            matched_by=matched_by,
            learned_plate=learned_plate,
        )
        return winner, False


def revive_claim(session: Session, row: EventIdentity) -> tuple[EventIdentity, bool]:
    """Un-repudiate `row`, unless a racing writer already claimed the pair.

    Returns (surviving_row, revived). When the pair has since acquired a
    standing claim, `row` is left unlinked — the repudiation stays on the
    record — and the existing claim is returned with revived=False.
    """
    was_unlinked_at = row.unlinked_at
    try:
        with session.begin_nested():
            row.unlinked_at = None
            session.flush()
        return row, True
    except IntegrityError:
        # Restore before querying: an autoflush would otherwise re-issue
        # the UPDATE that just failed.
        row.unlinked_at = was_unlinked_at
        winner = active_claim(session, row.event_id, row.identity_id)
        if winner is None:
            raise
        return winner, False


def link_claim(
    session: Session,
    *,
    event_id: int,
    identity_id: int,
    identifier_key: str | None,
    similarity: float,
    matched_by: str | None,
    learned_plate: bool = False,
) -> tuple[EventIdentity, bool]:
    """Claim this identity on this event, or record another sighting.

    Returns (link, created). An *unlinked* row is never revived by a
    later frame (CLD-36): a matching frame makes a fresh claim and leaves
    the operator's correction intact for review.
    """
    link = active_claim(session, event_id, identity_id)
    if link is not None:
        record_sighting(
            link,
            similarity=similarity,
            matched_by=matched_by,
            learned_plate=learned_plate,
        )
        return link, False
    return insert_claim(
        session,
        EventIdentity(
            event_id=event_id,
            identity_id=identity_id,
            identifier_key=identifier_key,
            similarity=similarity,
            matched_by=matched_by,
            learned_plate=learned_plate,
        ),
    )


def repair_duplicate_claims(session: Session) -> int:
    """Collapse pre-index duplicate active claims. Returns rows folded.

    Runs from init_db before the partial unique index is created, over
    databases written while nothing forbade the duplicates (CLD-133: ~10%
    of the live link rows). Idempotent and resumable in the shape
    CLAUDE.md asks of every long operation: the HAVING query *is* the
    skip-what's-done query, since a repaired group stops matching it, and
    the commit is per group, so an interruption leaves whole groups done
    and the rest re-found on the next boot.
    """
    groups = session.execute(
        select(EventIdentity.event_id, EventIdentity.identity_id)
        .where(
            EventIdentity.identity_id.is_not(None),
            EventIdentity.unlinked_at.is_(None),
        )
        .group_by(EventIdentity.event_id, EventIdentity.identity_id)
        .having(func.count() > 1)
    ).all()
    folded = 0
    for event_id, identity_id in groups:
        rows = session.scalars(
            select(EventIdentity)
            .where(
                EventIdentity.event_id == event_id,
                EventIdentity.identity_id == identity_id,
                EventIdentity.unlinked_at.is_(None),
            )
            # The row with the most evidence behind it survives; on a tie
            # the oldest, which is the one an operator may already have
            # reviewed or linked to.
            .order_by(
                EventIdentity.hit_count.desc(),
                EventIdentity.similarity.desc(),
                EventIdentity.id,
            )
        ).all()
        keeper, losers = rows[0], rows[1:]
        for loser in losers:
            fold_claim(keeper, loser)
            session.delete(loser)
            folded += 1
        session.commit()
    return folded
