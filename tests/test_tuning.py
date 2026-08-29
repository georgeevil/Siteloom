"""The detector tuning core (siteloom/tuning.py).

Trials run with an injected stub module — no model weights — over the
synthetic sample video; the pure pieces (presets, the merge↔split axis,
overrides, recommendations, report comparison) are tested on dicts with
known answers.
"""

from __future__ import annotations

import json

from siteloom.config import DetectionConfig
from siteloom.tuning import (
    AXIS,
    PRESETS,
    friendly_error,
    plain_comparison,
    plain_summary,
    apply_overrides,
    axis_overrides,
    compare_reports,
    recommend,
    run_trial,
)


class ScriptedModule:
    """Replays scripted per-frame detections, like the ingest tests'
    SequenceDetector."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def process(self, job):
        dets = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return {"detections": [dict(d) for d in dets]}


def det(track_id, bbox=(100.0, 100.0, 200.0, 350.0), cls="person", conf=0.9):
    return {"class_name": cls, "confidence": conf, "bbox": list(bbox),
            "track_id": track_id, "zones": []}


# -- presets and the axis --------------------------------------------------


def test_presets_say_whether_they_are_tested():
    """Only one camera has a corpus; a preset must not imply otherwise."""
    assert PRESETS["site-defaults"]["tested"] is True
    untested = [k for k, p in PRESETS.items() if not p["tested"]]
    assert untested  # the scene bundles are honest starting points


def test_the_axis_moves_the_three_knobs_together():
    """CLD-102: `track_buffer`, `match_thresh` and `new_track_thresh`
    exposed separately guarantee nonsensical combinations — the axis is
    one slider between two named failures."""
    for position in (-2, -1, 1, 2):
        o = axis_overrides(position)
        assert set(o) == {"track_buffer_s", "tracker"}
        assert set(o["tracker"]) == {"match_thresh", "new_track_thresh"}
    assert axis_overrides(0) == {}
    # Monotonic: merging more = longer buffer, laxer matching.
    buffers = [AXIS[p].get("track_buffer_s", 4.0) for p in (-2, -1, 0, 1, 2)]
    assert buffers == sorted(buffers)


def test_apply_overrides_merges_tracker_and_replaces_the_rest():
    base = DetectionConfig(confidence=0.4,
                           tracker={"fuse_score": False, "match_thresh": 0.8})
    eff = apply_overrides(base, {
        "confidence": 0.6, "tracker": {"match_thresh": 0.9},
    })
    assert eff.confidence == 0.6
    assert eff.tracker == {"fuse_score": False, "match_thresh": 0.9}
    assert base.confidence == 0.4  # base untouched


# -- the trial -------------------------------------------------------------


def test_a_trial_writes_report_and_evidence(sample_video, tmp_path):
    frames = [[det(1)], [det(1)], [det(1), det(2, (600.0, 100.0, 700.0, 350.0))]]
    frames += [[det(1), det(2, (600.0, 100.0, 700.0, 350.0))]] * 7
    report = run_trial(
        sample_video, DetectionConfig(), 5.0, tmp_path / "run",
        module=ScriptedModule(frames),
    )
    assert report["frames"] == 10
    assert report["groups"]["person"]["tracks"] == 2
    # Births lead the evidence: track 1's first frame and track 2's.
    births = [m for m in report["moments"] if m["kind"] == "birth"]
    assert len(births) == 2
    on_disk = json.loads((tmp_path / "run" / "report.json").read_text())
    assert on_disk["groups"] == report["groups"]
    for m in report["moments"]:
        assert (tmp_path / "run" / m["file"]).is_file()


def test_a_trial_groups_metrics_by_class_group(sample_video, tmp_path):
    """People and cars are different populations — pooling their box
    widths and track counts would answer a question nobody asked."""
    frames = [[det(1), det(9, (400.0, 200.0, 900.0, 420.0), cls="car")]] * 10
    report = run_trial(
        sample_video, DetectionConfig(), 5.0, tmp_path / "run",
        module=ScriptedModule(frames),
        group_for=lambda cls: ["car", "truck"] if cls == "car" else [cls],
    )
    assert set(report["groups"]) == {"person", "car|truck"}
    assert report["groups"]["person"]["tracks"] == 1


def test_the_sample_video_reads_as_color_not_ir(sample_video, tmp_path):
    report = run_trial(
        sample_video, DetectionConfig(), 5.0, tmp_path / "run",
        module=ScriptedModule([[det(1)]] * 10),
    )
    assert report["scene"]["ir"] is False  # the synthetic clip is colored


# -- recommendations -------------------------------------------------------


def group(tracks=2, obs=40, step_iou=0.9, births=(0, 0)):
    return {
        "tracks": tracks, "observations": obs, "detection_rate": 0.9,
        "median_step_iou": step_iou, "median_box_px": 120.0,
        "bridges": 0, "implausible_bridges": 0, "crossings": 0,
        "mid_occlusion_births": births[0], "post_occlusion_births": births[1],
    }


def report_with(groups, ir=False, classes=None):
    return {
        "groups": groups, "sample_fps": 2.0,
        "scene": {"ir": ir, "saturation_mean": 5.0 if ir else 60.0,
                  "classes": classes or {}},
    }


def fields(recs):
    return [r["field"] for r in recs]


def test_coarse_sampling_of_people_recommends_faster_sampling():
    recs = recommend(report_with({"person": group(step_iou=0.3)}), 2.0)
    fps = next(r for r in recs if r["field"] == "sample_fps")
    assert fps["suggested"] == 5.0
    assert fps["basis"] == "measured"  # the CLD-5 rule is corpus-backed


def test_fast_sampling_gets_no_fps_suggestion():
    recs = recommend(report_with({"person": group(step_iou=0.9)}), 5.0)
    assert "sample_fps" not in fields(recs)


def test_ir_footage_warns_about_appearance_reid():
    recs = recommend(report_with({"person": group()}, ir=True), 5.0)
    assert any("IR" in r["reason"] for r in recs)


def test_track_churn_suggests_a_class_floor():
    classes = {"person": {"count": 30, "median_width_px": 100.0,
                          "median_confidence": 0.55, "p25_confidence": 0.42}}
    recs = recommend(
        report_with({"person": group(tracks=12, obs=20)}, classes=classes), 5.0
    )
    churn = next(r for r in recs if r["field"] == "class_confidence.person")
    assert churn["suggested"] == 0.55
    assert churn["basis"] == "heuristic"  # honest about not being measured


def test_occlusion_births_point_at_the_corpus():
    recs = recommend(report_with({"person": group(births=(1, 1))}), 5.0)
    assert any("corpus" in r["reason"] for r in recs)


def test_an_empty_trial_says_so():
    recs = recommend({"groups": {}, "scene": {"classes": {}}}, 5.0)
    assert any("Nothing was tracked" in r["reason"] for r in recs)


# -- comparing persisted reports -------------------------------------------


def test_compare_reports_uses_the_harness_verdict():
    base = {"groups": {"person": group(tracks=8, births=(1, 1))}}
    cand = {"groups": {"person": group(tracks=8, births=(0, 0))}}
    out = compare_reports(base, cand)
    assert out["person"]["verdict"] == "better"
    fragmented = {"groups": {"person": group(tracks=25)}}
    assert compare_reports(base, fragmented)["person"]["verdict"].startswith(
        "rejected"
    )


# -- operator-facing words -------------------------------------------------


def test_the_nvr_export_failure_gets_guidance():
    raw = ("BadRequest: Request failed: https://192.168.1.77/proxy/protect/"
           "api/video/export?camera=x&start=1&end=2&channel=0 - "
           "Status: 404 - Reason: 502")
    words = friendly_error(raw)
    assert words is not None
    assert "Upload" in words  # the workaround, not just the diagnosis


def test_unknown_failures_stay_unknown():
    assert friendly_error("ZeroDivisionError: division by zero") is None


def test_unreachable_nvr_names_the_host_setting():
    assert "unifi" in friendly_error("ClientConnectorError: Connection refused")


def test_plain_summary_reads_like_a_sentence():
    words = plain_summary(report_with(
        {"person": group(tracks=2), "car|truck": group(tracks=1)},
    ))
    assert "2 people" in words
    assert "1 vehicle" in words
    assert "no tracking problems" in words


def test_plain_summary_names_the_failures_not_the_metrics():
    r = report_with({"person": group(births=(1, 1))})
    r["groups"]["person"]["implausible_bridges"] = 1
    words = plain_summary(r)
    assert "merged" in words and "split" in words
    assert "bridge" not in words  # jargon stays out of the sentence


def test_plain_summary_flags_ir_and_emptiness():
    assert "night (IR)" in plain_summary(
        report_with({"person": group()}, ir=True)
    )
    empty = plain_summary({"groups": {}, "scene": {}, "sample_fps": 5.0,
                           "frames": 50})
    assert "Nothing was detected" in empty


def test_plain_comparison_translates_verdicts():
    words = plain_comparison({
        "person": {"verdict": "better", "tracks": (3, 2),
                   "switch_like": (2, 0)},
        "car|truck": {"verdict": "rejected: bought it with fragmentation",
                      "tracks": (2, 9), "switch_like": (0, 0)},
    })
    assert "does better" in words
    assert "splitting subjects into more tracks" in words
