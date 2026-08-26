"""Scene condition measurement — the day/night signal (CLD-129).

IR footage is the condition that changes what good detection settings
are: color washes out, appearance re-ID loses the signal it leans on,
and the tracker bundle that wins in daylight is not the one that wins at
night. The measurement is cheap and direct — mean HSV saturation, the
same arithmetic the tuning lab's trial reports use (`scene.ir`) — and it
is measured from the *footage*, never the clock: reindex and backfill
process yesterday's frames at today's wall time, and a camera under a
floodlight never goes IR at all.

`ProfileMonitor` turns the per-frame measurement into a per-camera
day/night decision with hysteresis, because the switch is expensive:
ultralytics reads tracker settings once per predictor, so a profile
change means a fresh model instance and a track-id restart (absorbed by
the CLD-40 stitch/merge layers, but not free). Dusk holds saturation
near the threshold for minutes — the confirm-streak and the dwell floor
are what keep that from thrashing.

Decisions are in frame time and deterministic from the footage; the
monitor object itself is in-memory advisory state, the same accepted
class as ingest's OcclusionMonitor (CLD-305) and plate-ration notes — a
restart re-derives the profile within `CONFIRM_FRAMES` frames.
"""

from __future__ import annotations

#: Mean HSV saturation below this reads as IR/grayscale footage. Shared
#: with the tuning lab's trial reports — one threshold, so a trial's
#: "reads as night (IR)" and live profile switching cannot disagree.
IR_SATURATION = 14.0

#: Consecutive frames past the threshold before the profile flips —
#: 3 s at the live cameras' 5 fps. A headlight sweep or a white wall
#: filling the frame is a frame or two, not a streak.
CONFIRM_FRAMES = 15

#: Once flipped, the profile holds at least this long (frame time).
#: Dusk sits near the threshold for minutes; every flip restarts the
#: camera's track ids, so flapping is the failure mode.
MIN_DWELL_S = 60.0


def mean_saturation(bgr) -> float:
    """Mean HSV saturation of a frame — the IR signal."""
    import cv2

    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


class ProfileMonitor:
    """Per-camera day/night state over frame-time saturation readings.

    `observe(t, saturation)` returns the profile in force for that
    frame — flips only after `confirm_frames` consecutive readings on
    the other side of the threshold, and never sooner than
    `min_dwell_s` after the previous flip. Starts in "day": an IR clip's
    opening frames run the day profile for the confirm streak, which is
    the causality cost hysteresis always pays.
    """

    def __init__(
        self,
        *,
        ir_saturation: float = IR_SATURATION,
        confirm_frames: int = CONFIRM_FRAMES,
        min_dwell_s: float = MIN_DWELL_S,
    ) -> None:
        self.ir_saturation = ir_saturation
        self.confirm_frames = confirm_frames
        self.min_dwell_s = min_dwell_s
        self.profile = "day"
        self._streak = 0
        self._switched_at: float | None = None
        self._last_t: float | None = None

    def observe(self, t: float, saturation: float) -> str:
        if self._last_t is not None and t < self._last_t - 1.0:
            # Time went backwards past jitter: an out-of-order backfill
            # seam. The streak is meaningless across it; the profile
            # itself stands until the new segment's own evidence flips
            # it.
            self._streak = 0
            self._switched_at = None
        self._last_t = t

        wants = "night" if saturation < self.ir_saturation else "day"
        if wants == self.profile:
            self._streak = 0
            return self.profile
        self._streak += 1
        if self._streak >= self.confirm_frames and (
            self._switched_at is None
            or t - self._switched_at >= self.min_dwell_s
        ):
            self.profile = wants
            self._streak = 0
            self._switched_at = t
        return self.profile
