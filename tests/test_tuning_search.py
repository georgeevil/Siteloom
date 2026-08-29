"""The propose-only settings search (CLD-102).

Scripted trial reports, no model weights. The properties under test are
the ticket's: bounded named candidates (never a grid), the verdict
arithmetic as the anti-gaming defence (a candidate that fragments its
way to zero switches loses), halving that spends clips on survivors,
interruptibility that never leaves a half-claim, and a null winner as a
finding rather than a failure.
"""

from __future__ import annotations

import pytest

from siteloom.config import DetectionConfig
from siteloom.tuning_search import candidates, clip_score, successive_halving


def group(tracks=4, births=0, bridges=0):
    return {
        "tracks": tracks, "observations": tracks * 10, "detection_rate": 0.9,
        "median_step_iou": 0.85, "median_box_px": 100.0,
        "bridges": bridges, "implausible_bridges": bridges, "crossings": 2,
        "mid_occlusion_births": births, "post_occlusion_births": 0,
    }


def report(tracks=4, births=0, bridges=0):
    return {"groups": {"person": group(tracks, births, bridges)}}


def test_candidates_are_named_bounded_and_deduped():
    effective = DetectionConfig()
    cands = candidates(effective)
    assert 0 < len(cands) <= 12
    assert all(":" in c["name"] for c in cands)  # named bundles, no grids
    # Nothing that resolves to what the camera already runs.
    from siteloom.tuning import apply_overrides

    for c in cands:
        assert (
            apply_overrides(effective, c["overrides"]).model_dump()
            != effective.model_dump()
        )


def test_fragmenting_to_zero_switches_scores_as_a_loss():
    """CLD-97's lesson as arithmetic: match_thresh 0.5 zeroed the
    bridges by quadrupling the tracks. The search must never crown it."""
    baseline = report(tracks=4, births=2)
    cheat = report(tracks=18, births=0)
    honest = report(tracks=4, births=0)
    assert clip_score(baseline, cheat) < 0
    assert clip_score(baseline, honest) > 0


def make_runner(outcomes):
    """outcomes: name -> report; baseline tag '' uses 'baseline'."""
    calls = []

    def run(clip, overrides, tag):
        calls.append((str(clip), tag))
        key = tag if tag in outcomes else "baseline"
        return f"run-{len(calls)}", outcomes[key]

    run.calls = calls
    return run


def test_halving_spends_clips_on_survivors_and_crowns_a_winner():
    outcomes = {
        "baseline": report(births=2),
        "good": report(births=0),
        "meh": report(births=2),
        "bad": report(births=4),
        "cheat": report(tracks=18, births=0),
    }
    cands = [{"name": n, "overrides": {"confidence": i / 10}}
             for i, n in enumerate(["good", "meh", "bad", "cheat"], start=1)]
    run = make_runner(outcomes)
    proposal = successive_halving(
        ["clip-a", "clip-b", "clip-c"], cands, run
    )
    assert proposal["winner"]["name"] == "good"
    assert proposal["winner"]["total"] > 0
    # The loser never saw the later clips.
    assert ("clip-c", "bad") not in run.calls
    assert ("clip-c", "cheat") not in run.calls
    # Rounds are recorded for the review page (three clips fit in two
    # rounds: everyone on the first, survivors on the rest).
    assert [r["round"] for r in proposal["rounds"]] == [1, 2]


def test_nothing_better_is_a_finding_not_a_failure():
    outcomes = {"baseline": report(births=0), "same": report(births=0)}
    proposal = successive_halving(
        ["clip-a"], [{"name": "same", "overrides": {"confidence": 0.5}}],
        make_runner(outcomes),
    )
    assert proposal["winner"] is None
    assert "finding" in proposal["verdict"]


def test_an_interrupt_escapes_between_trials():
    """`jobs cancel` reaches the search through check_interrupt — and
    the caller writes no proposal file on the way out (the route only
    writes after successive_halving returns)."""
    outcomes = {"baseline": report(), "x": report(births=0)}
    run = make_runner(outcomes)
    trips = {"n": 0}

    def tripwire():
        trips["n"] += 1
        if trips["n"] >= 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        successive_halving(
            ["clip-a", "clip-b"],
            [{"name": "x", "overrides": {"confidence": 0.5}}],
            run, check_interrupt=tripwire,
        )
    assert len(run.calls) <= 3  # it stopped mid-search, not at the end
