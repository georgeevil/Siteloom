"""Overnight settings search — proposes, never applies (CLD-102).

The search answers "is there a better bundle for this camera?" the only
way that survives CLD-102's analysis: a *bounded, named* candidate list
(the scene presets, the merge↔split axis, a couple of confidence
nudges — never a parameter grid), scored by the corpus's own verdict
arithmetic against a baseline of the camera's current settings. That
verdict is also the anti-gaming defence: a candidate that zeroes
switches by fragmenting is "rejected", which is precisely how
`match_thresh: 0.5` was caught by hand in CLD-97 — the guard is
reused, not reinvented.

Successive halving spends the compute where it matters: every candidate
runs on one clip, survivors earn more clips, and only finalists see the
whole set. Between every trial the reporter's interrupt is honoured —
`jobs cancel` works — and the proposal file is written only at the end:
an interrupted search leaves trials (reusable evidence) but never a
half-claim.

The proposal's evidence is the trials themselves: each candidate×clip
run is an ordinary tuning-lab run with its annotated frames, so the
review page shows crops, not just numbers — CLD-102's through-line.
Pure and injectable: `run` is the trial runner, so tests search with
scripted reports and no model weights.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from siteloom.config import DetectionConfig
from siteloom.tuning import AXIS, PRESETS, apply_overrides, compare_reports

log = logging.getLogger(__name__)

#: How the per-group verdicts fold into one clip score. Worst wins:
#: a candidate that helps people but fragments vehicles is not a win.
_VERDICT_SCORE = {
    "better": 1,
    "no change": 0,
    "only in one run": 0,
    "worse": -2,
    "rejected: bought it with fragmentation": -3,
}


def candidates(effective: DetectionConfig) -> list[dict[str, Any]]:
    """The bounded candidate list for one camera, deduped against what
    it already runs. Named bundles only — the search space is the space
    of *describable* configurations."""
    current = effective.model_dump()
    out: list[dict[str, Any]] = []

    def offer(name: str, overrides: dict[str, Any]) -> None:
        if apply_overrides(effective, overrides).model_dump() == current:
            return  # already what the camera runs
        if any(c["overrides"] == overrides for c in out):
            return
        out.append({"name": name, "overrides": overrides})

    for key, preset in PRESETS.items():
        if key == "site-defaults":
            continue
        overrides = {
            k: v for k, v in preset["settings"].items() if k != "sample_fps"
        }
        offer(f"preset:{key}", overrides)
    for position in (-2, -1, 1, 2):
        offer(f"axis:{position:+d}", AXIS[position])
    for delta in (-0.1, 0.1):
        floor = round(min(0.8, max(0.2, effective.confidence + delta)), 2)
        if floor != effective.confidence:
            offer(f"confidence:{floor}", {"confidence": floor})
    return out[:12]


def clip_score(baseline: dict, candidate: dict) -> int:
    """One clip's verdict, folded worst-first."""
    comparison = compare_reports(baseline, candidate)
    if not comparison:
        return 0
    return min(
        _VERDICT_SCORE.get(v["verdict"], 0) for v in comparison.values()
    )


def successive_halving(
    clips: list[Any],
    cands: list[dict[str, Any]],
    run: Callable[[Any, dict[str, Any], str], tuple[str, dict]],
    *,
    check_interrupt: Callable[[], None] = lambda: None,
    log_round: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """The search. `run(clip, overrides, tag)` executes one trial and
    returns (run_id, report). Returns the proposal dict — with a null
    winner when nothing beat the baseline, which is a finding, not a
    failure."""
    baselines: dict[Any, tuple[str, dict]] = {}

    def baseline_for(clip) -> tuple[str, dict]:
        if clip not in baselines:
            baselines[clip] = run(clip, {}, "baseline")
            check_interrupt()
        return baselines[clip]

    standings: list[dict[str, Any]] = [
        {"candidate": c, "scores": {}, "trials": {}} for c in cands
    ]
    rounds: list[dict[str, Any]] = []
    # Round sizes: everyone on one clip, half on up to two more, the
    # finalists on everything. Clips come newest-first from the caller.
    schedule = [clips[:1], clips[1:3], clips[3:]]
    for round_no, round_clips in enumerate(schedule, start=1):
        if not standings or not round_clips:
            continue
        for clip in round_clips:
            base_id, base_report = baseline_for(clip)
            for entry in standings:
                run_id, report = run(
                    clip, entry["candidate"]["overrides"],
                    entry["candidate"]["name"],
                )
                entry["scores"][str(clip)] = clip_score(base_report, report)
                entry["trials"][str(clip)] = run_id
                check_interrupt()
        standings.sort(
            key=lambda e: sum(e["scores"].values()), reverse=True
        )
        keep = max(1, len(standings) // 2)
        cut = standings[keep:]
        rounds.append({
            "round": round_no,
            "clips": [str(c) for c in round_clips],
            "standings": [
                {
                    "name": e["candidate"]["name"],
                    "total": sum(e["scores"].values()),
                    "kept": e not in cut,
                }
                for e in standings
            ],
        })
        log_round(
            f"round {round_no}: kept {keep} of {len(standings)} candidates"
        )
        standings = standings[:keep]

    winner = standings[0] if standings else None
    total = sum(winner["scores"].values()) if winner else 0
    proposal: dict[str, Any] = {
        "clips": [str(c) for c in clips],
        "baseline_trials": {str(c): rid for c, (rid, _) in baselines.items()},
        "rounds": rounds,
        "winner": None,
    }
    if winner is not None and total > 0:
        proposal["winner"] = {
            "name": winner["candidate"]["name"],
            "overrides": winner["candidate"]["overrides"],
            "total": total,
            "scores": winner["scores"],
            "trials": winner["trials"],
        }
    else:
        proposal["verdict"] = (
            "nothing beat the current settings on these clips — that is a "
            "finding: the camera is as tuned as this clip set can show"
        )
    return proposal
