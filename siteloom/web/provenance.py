"""How a claim's evidence renders: badge, score, threshold (CLD-136).

A claim row showed `sim 1.00` for a plate match and `sim 0.00` for a
human link — the two least honest numbers on the console. Neither is a
similarity: the plate 1.0 is a sentinel `_match_plate` returns so a plate
outranks any cosine (PRD §6.4), and the human 0.0 is the absence of a
measurement, written because an operator's assertion has no score.
Rendered as decimals beside genuine ones, both read as confidence the
system does not have.

The rule lives here rather than in a template because it is where the
bugs are: a branch over a column two other modules mutate, plus a
threshold lookup with three outcomes. The markup it feeds is
`templates/_claims.html`.
"""

from __future__ import annotations

from dataclasses import dataclass

from siteloom.store.models import EventIdentity

#: Badge text per stored `matched_by`, and what the badge asserts.
#:
#: The column is *strongest evidence*, not first contact: `_absorb_evidence`
#: upgrades it on every later frame and `fold_claim` on every merge, both
#: by `_MATCH_RANK` (store/claims.py). A row that minted an identity
#: (None) and was later re-matched reads "visual"; any row a plate ever
#: touched reads "plate". Nothing persists how a claim was *first* made,
#: so the badge must not imply it — hence the wording of every title
#: below. First-contact provenance would need its own column.
_BADGES: dict[str | None, tuple[str, str]] = {
    "plate": ("plate", "matched by plate — strongest evidence recorded on this claim"),
    "visual": ("visual", "matched by appearance — strongest evidence recorded on this claim"),
    # The column says "human"; the operator's word is "manual". Two
    # spellings of one thing, deliberately: renaming the column would
    # move data that `siteloom/stats.py` reads.
    "human": ("manual", "linked by an operator — strongest evidence recorded on this claim"),
    None: ("new", "this claim created the identity; nothing was matched"),
}

#: What a score means on a `human` row that carries one. Not
#: self-evident, and the composition is real — see `claim_display`.
_MANUAL_SCORE_TITLE = (
    "strongest visual evidence recorded on this claim; "
    "the link itself was made by an operator"
)


@dataclass(frozen=True)
class ClaimDisplay:
    badge: str
    title: str
    #: None when there is nothing honest to show — never 0.0 as a stand-in.
    score: float | None
    #: The bar in force *now*, or None when nothing is configured.
    threshold: float | None
    #: Set only when the score needs explaining beside its badge.
    score_title: str | None = None


def claim_display(link: EventIdentity, threshold: float | None) -> ClaimDisplay:
    """What this claim's evidence honestly says.

    A score is shown when it is a measurement of something:

    * **Never on a plate row.** `_match_plate` returns a hardcoded 1.0 and
      `_absorb_evidence` keeps `max(existing, incoming)`, so once a plate
      match touches a row its `similarity` is pinned at the sentinel and
      every later genuine cosine is swallowed. The field is not a stale
      similarity; it is not a similarity at all, and no later frame can
      make it one again.
    * **Not at 0.0**, which is how `_attach` records "an operator said so,
      and no machine measured anything".
    * **Yes on a `manual` row above 0.0**, which looks contradictory and
      is not. `_attach` inserts `similarity=0.0, matched_by="human"`; if
      an ingest frame arrives afterwards, `record_sighting` raises the
      score to the real cosine while `matched_by` stays "human", because
      `_MATCH_RANK` ranks "visual" and "human" equally and only a
      strictly greater rank wins. The number is a true measurement on a
      claim a person made, and keying the score to the badge would hide
      it on exactly the rows an operator is most likely auditing.

    `threshold` is the effective bar from `IdentityConfig.threshold_for`
    — per-camera override, then site-wide, then None for an auto-added
    class whose default lives in the registry. None renders the score
    bare: inventing a bar would be the same kind of lie as the 1.00.
    """
    badge, title = _BADGES.get(link.matched_by, (link.matched_by or "", ""))
    show_score = link.matched_by != "plate" and (link.similarity or 0.0) > 0
    return ClaimDisplay(
        badge=badge,
        title=title,
        score=link.similarity if show_score else None,
        threshold=threshold if show_score else None,
        score_title=(
            _MANUAL_SCORE_TITLE if show_score and link.matched_by == "human" else None
        ),
    )
