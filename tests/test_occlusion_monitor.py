"""The streaming occlusion monitor (siteloom/tracking/occlusion.py).

Causal counterpart of the corpus's offline occlusion metrics, fed
synthetic frames with a known answer — same style as test_track_eval,
whose offline functions share this module's arithmetic by construction
(track_eval imports them from here).
"""

from __future__ import annotations

from siteloom.tracking.occlusion import OcclusionMonitor


def box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def det(track_id, bbox):
    return {"track_id": track_id, "bbox": bbox}


BIG = 200.0, 400.0   # the occluder
SMALL = 60.0, 120.0  # the hidden subject's partial box


def overlapped_frame(i, cx=500.0):
    """Track 1's big box with track 2's small box inside it."""
    return [
        det(1, box(cx + i, 300.0, *BIG)),
        det(2, box(cx + i + 20, 320.0, *SMALL)),
    ]


def separated_frame(i):
    return [
        det(1, box(200.0 + i, 300.0, *BIG)),
        det(2, box(900.0 + i, 300.0, *BIG)),
    ]


def feed_frames(monitor, frames, t0=0.0, dt=0.2):
    out = []
    for i, dets in enumerate(frames):
        out.append(monitor.feed(t0 + i * dt, dets))
    return out


def test_separated_subjects_are_never_annotated():
    notes = feed_frames(OcclusionMonitor(), [separated_frame(i) for i in range(10)])
    for frame in notes:
        for n in frame:
            assert n == {"occluded_with": [], "suspect_birth": None}


def test_sustained_containment_marks_both_participants():
    notes = feed_frames(OcclusionMonitor(), [overlapped_frame(i) for i in range(5)])
    # Frame 0 cannot know yet — the episode confirms on the second high
    # frame — and from then on both detections carry the other's id.
    assert notes[0][0]["occluded_with"] == []
    for frame in notes[1:]:
        assert frame[0]["occluded_with"] == [2]
        assert frame[1]["occluded_with"] == [1]


def test_a_track_born_inside_an_open_episode_is_a_mid_suspect():
    monitor = OcclusionMonitor()
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)])
    # A sliver mints track 3 inside the open episode's region.
    notes = monitor.feed(1.0, overlapped_frame(5)
                         + [det(3, box(510.0, 350.0, 40.0, 80.0))])
    suspect = notes[2]["suspect_birth"]
    assert suspect == {"candidates": [1, 2], "reason": "mid"}
    # The suspicion sticks to later frames, where the stitch decision
    # actually happens.
    notes = monitor.feed(1.2, overlapped_frame(6)
                         + [det(3, box(511.0, 350.0, 40.0, 80.0))])
    assert notes[2]["suspect_birth"]["reason"] == "mid"


def test_a_subject_first_seen_already_occluded_is_flagged_at_birth():
    """Track 2 only becomes visible at all once inside track 1's box.
    The flag must land on the birth frame — that is when the stitch
    decision happens, and a phantom's box overlaps the occluder, so a
    flag one frame late is a flag after the wrong stitch."""
    monitor = OcclusionMonitor()
    monitor.feed(0.0, [det(1, box(500.0, 300.0, *BIG))])
    monitor.feed(0.2, [det(1, box(501.0, 300.0, *BIG))])
    first = monitor.feed(0.4, overlapped_frame(2))
    assert first[1]["suspect_birth"] == {"candidates": [1], "reason": "mid"}
    # ... and sticks on later frames.
    confirmed = monitor.feed(0.6, overlapped_frame(3))
    assert confirmed[1]["suspect_birth"] == {"candidates": [1], "reason": "mid"}


def test_two_tracks_entering_already_overlapped_are_not_suspects():
    """Neither predates the other, so neither can be the other's phantom
    — two people can genuinely walk in shoulder to shoulder."""
    notes = feed_frames(OcclusionMonitor(), [overlapped_frame(i) for i in range(5)])
    for frame in notes:
        assert all(n["suspect_birth"] is None for n in frame)


def test_a_birth_near_a_closed_episode_is_a_post_suspect():
    monitor = OcclusionMonitor()
    monitor.feed(0.0, [det(1, box(500.0, 300.0, *BIG))])
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)], t0=0.2)
    # Both tracks vanish (subject fully hidden, then the pair is gone);
    # the episode closes on the co-presence gap. A stranger appears near
    # where it happened.
    monitor.feed(4.0, [])
    notes = monitor.feed(4.2, [det(9, box(520.0, 320.0, *SMALL))])
    assert notes[0]["suspect_birth"] == {"candidates": [1, 2], "reason": "post"}


def test_a_birth_during_the_close_gap_is_still_a_post_suspect():
    """The designed case: the hidden subject re-emerges *during* the
    close gap, while the pair is still nominally open — near the
    episode but outside its frozen region (the occluder kept moving).
    Suspicion is assigned only at birth, so a blind spot here would be
    permanent."""
    monitor = OcclusionMonitor()
    monitor.feed(0.0, [det(1, box(500.0, 300.0, *BIG))])
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)], t0=0.2)
    # Both tracks vanish; 1.0s later — inside the 2.0s close gap, the
    # episode not yet closed — a stranger appears half a box width past
    # the region's edge.
    notes = monitor.feed(2.0, [det(9, box(790.0, 320.0, *SMALL))])
    assert notes[0]["suspect_birth"] == {"candidates": [1, 2], "reason": "post"}


def test_a_birth_far_from_or_long_after_an_episode_is_ordinary():
    monitor = OcclusionMonitor()
    monitor.feed(0.0, [det(1, box(500.0, 300.0, *BIG))])
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)], t0=0.2)
    monitor.feed(4.0, [])
    far = monitor.feed(4.2, [det(8, box(5000.0, 300.0, *SMALL))])
    assert far[0]["suspect_birth"] is None
    late = monitor.feed(9.0, [det(9, box(520.0, 320.0, *SMALL))])
    assert late[0]["suspect_birth"] is None


def test_untracked_detections_pass_through_unannotated():
    monitor = OcclusionMonitor()
    notes = monitor.feed(0.0, [det(None, box(500.0, 300.0, *BIG))])
    assert notes == [{"occluded_with": [], "suspect_birth": None}]


def test_state_stays_serializable():
    """The monitor may later run behind the dispatcher; its state must be
    plain containers and numbers all the way down (PRD §5)."""
    monitor = OcclusionMonitor()
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)])
    monitor.feed(4.0, [det(9, box(520.0, 320.0, *SMALL))])

    def plain(value):
        if isinstance(value, dict):
            return all(plain(k) and plain(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return all(plain(v) for v in value)
        return isinstance(value, (str, int, float, bool)) or value is None

    assert plain(vars(monitor))


# -- swap evidence ---------------------------------------------------------

from siteloom.tracking.occlusion import swap_evidence  # noqa: E402

A = [1.0, 0.0]
B = [0.0, 1.0]


def test_a_clean_reappearance_is_not_a_swap():
    assert swap_evidence(
        post_a=[A], pre_a=[A], post_b=[B], pre_b=[B], min_margin=0.05
    ) is None


def test_crossed_appearances_are_a_swap():
    evidence = swap_evidence(
        post_a=[B], pre_a=[A], post_b=[A], pre_b=[B], min_margin=0.05
    )
    assert evidence is not None
    assert evidence["crossed"] == ["a", "b"]
    assert evidence["a_other"] > evidence["a_own"]


def test_one_side_crossing_is_enough():
    """When one subject leaves during the overlap, only one track
    survives to cross — that is still a swap."""
    evidence = swap_evidence(
        post_a=[B], pre_a=[A], post_b=[B], pre_b=[B], min_margin=0.05
    )
    assert evidence is not None
    assert evidence["crossed"] == ["a"]


def test_a_cross_match_inside_the_margin_is_not_evidence():
    """CLD-41's rule applied to the swap question: a suspicion must beat
    its runner-up, not merely edge past it."""
    nearly_a = [0.9, 0.4358]  # ~unit, sim to A ≈ 0.9
    assert swap_evidence(
        post_a=[nearly_a], pre_a=[A], post_b=[B], pre_b=[B], min_margin=0.5
    ) is None


def test_an_empty_side_is_no_verdict():
    """A side with no usable frames cannot cross; the other side is
    still judged on its own evidence."""
    assert swap_evidence(
        post_a=[], pre_a=[A], post_b=[B], pre_b=[B], min_margin=0.05
    ) is None


def test_closed_episodes_are_handed_over_exactly_once():
    monitor = OcclusionMonitor()
    feed_frames(monitor, [overlapped_frame(i) for i in range(5)])
    assert monitor.pop_closed() == []          # still open
    monitor.feed(1.0, separated_frame(0))
    monitor.feed(1.2, separated_frame(1))      # low streak closes it
    closed = monitor.pop_closed()
    assert len(closed) == 1
    assert closed[0]["tracks"] == [1, 2]
    assert closed[0]["t_start"] == 0.0
    assert monitor.pop_closed() == []          # drained
