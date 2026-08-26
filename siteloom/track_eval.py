"""Tracking quality metrics (CLD-98).

Tracker tuning trades two failures against each other, and moving one
knob always improves one of them:

* **ID switches** — one track absorbs two subjects. Every downstream
  claim on that event is then about the wrong person, which is how
  event 1392 produced fifteen wrong identities (CLD-97).
* **Fragmentation** — one subject becomes many tracks. Cheaper, because
  the stitcher exists to undo it, but it inflates event counts and
  starves each fragment of evidence.

Anyone can make either number zero by wrecking the other, so a
configuration is only meaningfully better if both are reported. That is
the whole reason this module exists as arithmetic rather than as a
paragraph in a PR.

Pure functions over observation timelines, deliberately holding no
opinion about where the observations came from — the harness feeds them
from a real detector over real footage, and the tests feed them
synthetic timelines with a known answer.

## What a "bridge" is, and why it stands in for an ID switch

Ground-truth ID switches need per-frame human labelling, which does not
exist and would not survive a config change. A bridge is the observable
that precedes one: a track that goes dark for longer than
`bridge_gap_s` and then resumes somewhere else has been re-acquired on a
coasting motion prediction rather than tracked. Not every bridge is a
switch — a subject can genuinely walk behind a tree and out again — but
every switch of this kind is a bridge, so the count is an upper bound on
the failure and the *distance* separates the plausible from the absurd.

Distance is reported in box widths, not pixels. 165 px means nothing
without knowing whether the subject is 40 px wide or 400; a jump of two
box widths is implausible for a person at any distance from any camera.

## Occlusion episodes, and why bridges cannot see them

A bridge needs a track to go dark. When one person walks *behind*
another, neither track goes dark — the occluder is detected throughout,
and the hidden person's sliver of arm keeps minting fresh partial-box
tracks. The clip that motivated this (backyard-puerta, two people
co-moving toward the camera) produced extra IDs during the overlap and
again when the hidden person stepped out, and the bridge count stayed
zero the whole time.

So the occlusion metrics watch box *containment* — intersection over the
smaller box's area — rather than IoU, because the hidden subject's
partial box is small relative to the occluder's and their IoU never gets
large. Two co-present tracks whose containment stays high for a few
frames open an `OcclusionEpisode`; a track first observed inside an open
episode's region is a `mid_occlusion_birth`, and one first observed just
after an episode closes, near where it happened, is a
`post_occlusion_birth`. Both are fragmentation the ordinary track count
underweights (each phantom lives a handful of frames) and both are where
identity claims go to die — a partial-box crop is the worst kind of
gallery evidence.

`crossings` (the episode count) is reported alongside, so a config that
zeroes the births by never overlapping boxes at all — which would mean
the detector stopped seeing the hidden person — is visible rather than
silently rewarded.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# The occlusion arithmetic is shared with the live pipeline's
# OcclusionMonitor (siteloom/tracking/occlusion.py) so the harness and
# ingest cannot drift: a config the corpus judged clean is judged by the
# same containment/episode logic ingest gates on. Re-exported here
# because this module is the harness's single import surface.
from siteloom.tracking.occlusion import (  # noqa: F401
    OcclusionEpisode,
    classify_births,
    containment,
    find_occlusions,
)


@dataclass(frozen=True)
class Observation:
    """One detection of one track at one moment."""

    track_id: int
    t: float  # seconds from the start of the clip
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass(frozen=True)
class Bridge:
    """A re-acquisition: one track, dark for a while, resuming elsewhere."""

    track_id: int
    gap_s: float
    jump_px: float
    #: Jump measured in box widths, which is comparable across cameras
    #: and subject distances in a way pixels are not.
    jump_widths: float

    @property
    def implausible(self) -> bool:
        """Further than a subject could plausibly have moved unobserved.

        One box width is the threshold because it is the point at which
        the old and new boxes cannot overlap at all — beyond it, nothing
        but the motion prediction connects them.
        """
        return self.jump_widths >= 1.0


@dataclass
class TrackingReport:
    """What one configuration did to one clip."""

    frames: int = 0
    frames_with_detection: int = 0
    observations: int = 0
    tracks: int = 0
    bridges: list[Bridge] = field(default_factory=list)
    occlusions: list[OcclusionEpisode] = field(default_factory=list)
    #: Track ids first observed inside an open episode's region — the
    #: sliver-of-arm phantom minted while its subject is hidden.
    mid_occlusion_births: list[int] = field(default_factory=list)
    #: Track ids first observed just after an episode closed, near where
    #: it happened — the hidden subject re-emerging as a stranger.
    post_occlusion_births: list[int] = field(default_factory=list)
    median_box_px: float = 0.0
    #: IoU between consecutive observations of the same track, when they
    #: really are consecutive. Low values mean the sample rate is too
    #: coarse for the subject's speed — a different problem from a
    #: bridge, and one this separates out rather than conflating.
    median_step_iou: float | None = None
    seconds: float = 0.0

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames_with_detection / self.frames

    @property
    def implausible_bridges(self) -> list[Bridge]:
        return [b for b in self.bridges if b.implausible]

    @property
    def crossings(self) -> int:
        """How often the clip actually put one box inside another. Zero
        births over zero crossings proves nothing; zero over three is a
        result."""
        return len(self.occlusions)

    @property
    def worst_bridge(self) -> Bridge | None:
        """The one most likely to be a switch — ranked by distance, not
        duration. A long gap the subject barely moved across is a
        successful re-acquisition; a short one they crossed the frame in
        is not."""
        return max(self.bridges, key=lambda b: b.jump_widths, default=None)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def summarize(
    observations: list[Observation],
    *,
    frames: int,
    frames_with_detection: int,
    sample_interval_s: float,
    bridge_gap_s: float = 2.0,
    step_tolerance: float = 1.5,
    seconds: float = 0.0,
    occl_containment: float = 0.5,
    occl_min_frames: int = 2,
    birth_window_s: float = 3.0,
    birth_radius_w: float = 1.0,
) -> TrackingReport:
    """Fold a clip's observations into one comparable report.

    `sample_interval_s` is the spacing between sampled frames — 1/5 s at
    5 fps. It is required rather than derived, because the only other
    thing to derive it from is `seconds`, which is wall-clock runtime;
    conflating the two silently produced an interval of ~40 ms against a
    real 200 ms and left step-IoU permanently empty.

    `step_tolerance` multiplies that interval to decide which consecutive
    pairs count as "no frames missing" for the step-IoU figure. Pairs
    wider than that are gaps, and measuring overlap across a gap would
    answer a question nobody asked.
    """
    report = TrackingReport(
        frames=frames,
        frames_with_detection=frames_with_detection,
        observations=len(observations),
        seconds=seconds,
    )
    if not observations:
        return report

    widths = [o.width for o in observations if o.width > 0]
    report.median_box_px = statistics.median(widths) if widths else 0.0

    by_track: dict[int, list[Observation]] = {}
    for o in observations:
        by_track.setdefault(o.track_id, []).append(o)
    report.tracks = len(by_track)

    # A per-track median width, so a bridge on a distant subject is
    # judged against that subject's scale rather than the scene's.
    step_ious: list[float] = []
    interval = sample_interval_s
    for track_id, obs in by_track.items():
        obs.sort(key=lambda o: o.t)
        track_widths = [o.width for o in obs if o.width > 0]
        scale = statistics.median(track_widths) if track_widths else report.median_box_px
        for prev, cur in zip(obs, obs[1:]):
            gap = cur.t - prev.t
            if gap > bridge_gap_s:
                (px, py), (cx, cy) = prev.centre, cur.centre
                jump = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                report.bridges.append(Bridge(
                    track_id=track_id,
                    gap_s=gap,
                    jump_px=jump,
                    jump_widths=(jump / scale) if scale > 0 else 0.0,
                ))
            elif interval and gap <= interval * step_tolerance:
                step_ious.append(iou(prev.bbox, cur.bbox))

    if step_ious:
        report.median_step_iou = statistics.median(step_ious)

    report.occlusions = find_occlusions(
        observations,
        occl_containment=occl_containment,
        occl_min_frames=occl_min_frames,
        close_gap_s=bridge_gap_s,
    )
    report.mid_occlusion_births, report.post_occlusion_births = classify_births(
        observations,
        report.occlusions,
        birth_window_s=birth_window_s,
        birth_radius_w=birth_radius_w,
        close_gap_s=bridge_gap_s,
    )
    return report


def compare(baseline: TrackingReport, candidate: TrackingReport) -> dict[str, object]:
    """Did the candidate actually win?

    A verdict rather than a diff, because the failure mode this whole
    module exists to prevent is reading one improved number and shipping.
    A candidate wins only by reducing implausible bridges *without*
    inflating track count — the two ways of cheating are removing every
    bridge by fragmenting, and removing fragmentation by letting tracks
    absorb everything.

    Occlusion births count on the same axis as implausible bridges: both
    are a subject acquiring an identity it should not have, and a config
    that trades one for the other has not improved anything.
    """

    def switch_like(r: TrackingReport) -> int:
        return (
            len(r.implausible_bridges)
            + len(r.mid_occlusion_births)
            + len(r.post_occlusion_births)
        )

    fewer_switches = switch_like(candidate) < switch_like(baseline)
    no_worse_switches = switch_like(candidate) <= switch_like(baseline)
    # 25% more tracks for the same footage is fragmentation, not nuance.
    fragmented = candidate.tracks > baseline.tracks * 1.25
    less_fragmented = candidate.tracks < baseline.tracks

    if fragmented:
        verdict = "rejected: bought it with fragmentation"
    elif fewer_switches or (no_worse_switches and less_fragmented):
        verdict = "better"
    elif no_worse_switches and candidate.tracks == baseline.tracks:
        verdict = "no change"
    else:
        verdict = "worse"

    return {
        "verdict": verdict,
        "implausible_bridges": (
            len(baseline.implausible_bridges), len(candidate.implausible_bridges)
        ),
        "occlusion_births": (
            len(baseline.mid_occlusion_births) + len(baseline.post_occlusion_births),
            len(candidate.mid_occlusion_births) + len(candidate.post_occlusion_births),
        ),
        "crossings": (baseline.crossings, candidate.crossings),
        "tracks": (baseline.tracks, candidate.tracks),
        "detection_rate": (baseline.detection_rate, candidate.detection_rate),
    }
