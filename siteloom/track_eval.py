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
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


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
    return report


def compare(baseline: TrackingReport, candidate: TrackingReport) -> dict[str, object]:
    """Did the candidate actually win?

    A verdict rather than a diff, because the failure mode this whole
    module exists to prevent is reading one improved number and shipping.
    A candidate wins only by reducing implausible bridges *without*
    inflating track count — the two ways of cheating are removing every
    bridge by fragmenting, and removing fragmentation by letting tracks
    absorb everything.
    """
    fewer_switches = len(candidate.implausible_bridges) < len(baseline.implausible_bridges)
    no_worse_switches = len(candidate.implausible_bridges) <= len(baseline.implausible_bridges)
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
        "tracks": (baseline.tracks, candidate.tracks),
        "detection_rate": (baseline.detection_rate, candidate.detection_rate),
    }
