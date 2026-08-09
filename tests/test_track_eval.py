"""Tracking quality metrics (CLD-98).

Fed synthetic timelines with a known answer, because the harness that
feeds these from a real detector needs model weights and real footage and
so cannot run here. What is testable is the arithmetic, and the arithmetic
is where a metric quietly starts lying.
"""

from __future__ import annotations

from siteloom.track_eval import Observation, compare, iou, summarize


def walk(track_id, start_t, n, x0, step=5.0, w=100.0, h=250.0, dt=0.2):
    """A subject moving steadily — the ordinary case."""
    return [
        Observation(track_id, start_t + i * dt,
                    (x0 + i * step, 100.0, x0 + i * step + w, 100.0 + h))
        for i in range(n)
    ]


def report(obs, **kw):
    kw.setdefault("frames", 100)
    kw.setdefault("frames_with_detection", 90)
    kw.setdefault("sample_interval_s", 0.2)   # matches walk()'s dt
    kw.setdefault("seconds", 20.0)
    return summarize(obs, **kw)


def test_continuous_tracking_has_no_bridges():
    r = report(walk(1, 0.0, 50, 200.0))
    assert r.tracks == 1
    assert r.bridges == []
    assert r.worst_bridge is None


def test_a_gap_the_subject_barely_moved_across_is_not_implausible():
    """Someone stands behind a tree and steps back out. The tracker did
    the right thing, and calling it a switch would make the metric
    useless."""
    obs = walk(1, 0.0, 10, 200.0) + walk(1, 6.0, 10, 210.0)
    r = report(obs)
    assert len(r.bridges) == 1
    bridge = r.bridges[0]
    assert bridge.gap_s > 2.0
    assert not bridge.implausible          # ~0.1 box widths
    assert r.implausible_bridges == []


def test_a_gap_the_subject_crossed_the_frame_in_is_implausible():
    """CLD-97's failure: 6.2s dark, then a box 1.7 widths away. Nothing
    but the motion prediction connects those."""
    obs = walk(1, 0.0, 10, 200.0) + walk(1, 8.0, 10, 420.0)
    r = report(obs)
    assert len(r.implausible_bridges) == 1
    assert r.implausible_bridges[0].jump_widths >= 1.0


def test_distance_is_measured_in_box_widths_not_pixels():
    """A 165px jump is damning for a 100px-wide subject and unremarkable
    for a 400px one. Pixels alone cannot be compared across cameras."""
    # Same centre displacement (200px) either side of the gap; only the
    # subject's size differs. step=0 so the walks do not add drift.
    near = report(walk(1, 0.0, 5, 200.0, step=0.0, w=400.0)
                  + walk(1, 8.0, 5, 400.0, step=0.0, w=400.0))
    far = report(walk(2, 0.0, 5, 200.0, step=0.0, w=100.0)
                 + walk(2, 8.0, 5, 400.0, step=0.0, w=100.0))
    assert near.bridges[0].jump_px == far.bridges[0].jump_px
    assert not near.bridges[0].implausible
    assert far.bridges[0].implausible


def test_each_track_is_judged_against_its_own_scale():
    """A distant subject in the same scene must not be measured against
    a close one's box."""
    obs = (walk(1, 0.0, 5, 100.0, step=0.0, w=400.0)
           + walk(1, 8.0, 5, 300.0, step=0.0, w=400.0)
           + walk(2, 0.0, 5, 900.0, step=0.0, w=60.0)
           + walk(2, 8.0, 5, 1000.0, step=0.0, w=60.0))
    r = report(obs)
    by_track = {b.track_id: b for b in r.bridges}
    assert not by_track[1].implausible     # 200px on a 400px subject
    assert by_track[2].implausible         # 100px on a 60px subject


def test_worst_bridge_ranks_by_distance_not_duration():
    """A long gap barely moved across is a success; a short one crossing
    the frame is not. Ranking by duration would surface the wrong one."""
    obs = (walk(1, 0.0, 5, 200.0, step=0.0) + walk(1, 30.0, 5, 205.0, step=0.0)
           + walk(2, 0.0, 5, 200.0, step=0.0) + walk(2, 3.0, 5, 500.0, step=0.0))
    r = report(obs)
    assert r.worst_bridge.track_id == 2


def test_step_iou_needs_the_sampling_interval_not_the_runtime():
    """Deriving the interval from wall-clock runtime gave ~40ms against a
    real 200ms, so every pair looked like a gap and step-IoU was silently
    always empty. Passing runtime here must not resurrect that."""
    obs = walk(1, 0.0, 10, 200.0)
    assert report(obs, sample_interval_s=0.2).median_step_iou is not None
    assert report(obs, sample_interval_s=0.04).median_step_iou is None


def test_step_iou_measures_only_genuinely_consecutive_frames():
    """Measuring overlap across a six-second hole answers a question
    nobody asked, and would drag the sample-rate signal to zero."""
    obs = walk(1, 0.0, 10, 200.0) + walk(1, 8.0, 10, 400.0)
    r = report(obs)
    # Steps within a track are 5px on a 100px box: near-total overlap.
    assert r.median_step_iou > 0.9


def test_no_observations_is_reported_not_crashed():
    r = report([])
    assert r.tracks == 0
    assert r.detection_rate == 0.9
    assert r.median_step_iou is None
    assert r.worst_bridge is None


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


# -- the verdict ----------------------------------------------------------


def make(tracks, implausible):
    """`tracks` distinct ids, `implausible` of which bridge a big jump.

    The bridging tracks are counted within `tracks`, not added to it —
    otherwise a candidate with more switches also looks fragmented and
    the two signals stop being separable.
    """
    obs = []
    for t in range(tracks):
        x = 200.0 + t * 500
        if t < implausible:
            obs += (walk(t, 0.0, 5, x, step=0.0)
                    + walk(t, 9.0, 5, x + 400.0, step=0.0))
        else:
            obs += walk(t, 0.0, 5, x, step=0.0)
    return report(obs)


def test_removing_switches_by_fragmenting_is_rejected():
    """The `match_thresh: 0.5` result from CLD-97 — zero bridges, four
    times the tracks. Reading only the bridge count would have shipped
    it."""
    baseline = make(tracks=8, implausible=1)
    candidate = make(tracks=34, implausible=0)
    assert compare(baseline, candidate)["verdict"].startswith("rejected")


def test_fewer_switches_at_similar_track_count_wins():
    baseline = make(tracks=14, implausible=2)
    candidate = make(tracks=8, implausible=0)
    assert compare(baseline, candidate)["verdict"] == "better"


def test_an_identical_result_is_no_change():
    """BoT-SORT ReID scored exactly like plain ByteTrack. That is a
    finding, and it must not read as an improvement."""
    baseline = make(tracks=8, implausible=1)
    candidate = make(tracks=8, implausible=1)
    assert compare(baseline, candidate)["verdict"] == "no change"


def test_more_switches_is_worse():
    baseline = make(tracks=8, implausible=0)
    candidate = make(tracks=8, implausible=3)
    assert compare(baseline, candidate)["verdict"] == "worse"
