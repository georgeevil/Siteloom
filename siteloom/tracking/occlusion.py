"""Occlusion episodes over tracker output — shared arithmetic + monitor.

When one subject walks behind another, neither track goes dark: the
occluder stays detected throughout and the hidden subject's slivers mint
fresh partial-box tracks. Every downstream failure that follows — a
phantom identity from a sliver crop, the hidden person re-emerging as a
stranger, a silent ID swap — starts with a stretch where one box sat
inside another. This module is the one place that recognises such a
stretch, in two forms:

* the **offline functions** (`find_occlusions`, `classify_births`) fold
  a complete clip timeline into episodes and birth verdicts — these are
  the tracker corpus's metrics (`siteloom/track_eval.py` re-exports
  them);
* the **`OcclusionMonitor`** answers the same questions *causally*, one
  frame at a time, for ingest — which frames are inside an episode, and
  which fresh track ids look like occlusion phantoms.

They share `containment` and the same thresholds so the harness and the
live pipeline cannot drift apart: a config the corpus judged clean is
judged by the same arithmetic that ingest gates on.

Containment — intersection over the *smaller* box's area — rather than
IoU, because the hidden subject's partial box is small relative to the
occluder's, so their IoU stays modest exactly when the occlusion is
total. Containment reads 1.0 there.

The monitor's state is dicts/lists/floats only, by the processing-module
rule (PRD §5): it may later run behind the dispatcher, and serializable
state is what keeps that door open.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Two co-present boxes count as occluding when the smaller is at least
#: this fraction inside the other.
DEFAULT_CONTAINMENT = 0.5
#: ... for at least this many sampled frames (a one-frame graze is a
#: crossing, not an occlusion).
DEFAULT_MIN_FRAMES = 2
#: A pair not co-present for longer than this closes its episode: the
#: interesting continuation of "fully hidden" is that one track has no
#: boxes at all, and an episode must not bridge into the pair's next
#: unrelated encounter.
DEFAULT_CLOSE_GAP_S = 2.0
#: A track first seen within this long after an episode closed ...
DEFAULT_BIRTH_WINDOW_S = 3.0
#: ... and within this many episode box-widths of its region, is a
#: suspect post-occlusion birth. Box widths, not pixels — the same rule
#: bridge jumps follow, for the same reason.
DEFAULT_BIRTH_RADIUS_W = 1.0


@dataclass(frozen=True)
class OcclusionEpisode:
    """A stretch where one track's box sat inside another's.

    `region` is the union of both participants' boxes over the episode,
    and `scale` is the median participant box width — the unit that
    post-episode birth distances are measured in.
    """

    track_a: int
    track_b: int
    t_start: float
    t_end: float
    region: tuple[float, float, float, float]
    scale: float


def containment(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the *smaller* box's area.

    The occlusion signal, where IoU is not: a hidden person's partial box
    is small relative to the occluder's, so their IoU stays modest even
    when the small box sits entirely inside the big one — exactly the
    frames this exists to notice. Containment reads 1.0 there.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(
        (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    )
    return inter / smaller if smaller > 0 else 0.0


def distance_to_region(
    point: tuple[float, float], region: tuple[float, float, float, float]
) -> float:
    """Distance from a point to a rectangle; 0.0 inside it."""
    px, py = point
    x1, y1, x2, y2 = region
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def _frame_key(t: float) -> float:
    # Two detections from the same sampled frame carry the same t, but it
    # may have been computed per track and floats disagree in the last
    # bit. Millisecond rounding groups frames; samples are tens of
    # milliseconds apart.
    return round(t, 3)


# -- offline: fold a complete timeline into episodes ----------------------


def find_occlusions(
    observations,
    *,
    occl_containment: float = DEFAULT_CONTAINMENT,
    occl_min_frames: int = DEFAULT_MIN_FRAMES,
    close_gap_s: float = DEFAULT_CLOSE_GAP_S,
) -> list[OcclusionEpisode]:
    """Stretches where one co-present track's box sat inside another's.

    `observations` is a list of objects with `.track_id`, `.t`, `.bbox`,
    `.width`, `.centre` (the harness's `Observation`). An episode opens
    when a pair of tracks detected in the same frames holds containment
    >= `occl_containment` for >= `occl_min_frames` of those frames, and
    closes when it drops below for that many frames — or when the pair
    stops being co-present for more than `close_gap_s`.
    """
    by_track: dict[int, dict[float, object]] = {}
    for o in observations:
        by_track.setdefault(o.track_id, {})[_frame_key(o.t)] = o

    episodes: list[OcclusionEpisode] = []
    tracks = sorted(by_track)
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            shared = sorted(set(by_track[a]) & set(by_track[b]))
            if len(shared) < occl_min_frames:
                continue
            run: list[tuple] = []
            low_streak = 0
            prev_t: float | None = None
            for t in shared:
                oa, ob = by_track[a][t], by_track[b][t]
                if prev_t is not None and oa.t - prev_t > close_gap_s:
                    _close_run(episodes, run, a, b, occl_min_frames)
                    low_streak = 0
                prev_t = oa.t
                if containment(oa.bbox, ob.bbox) >= occl_containment:
                    run.append((oa, ob))
                    low_streak = 0
                else:
                    low_streak += 1
                    if low_streak >= occl_min_frames:
                        _close_run(episodes, run, a, b, occl_min_frames)
            _close_run(episodes, run, a, b, occl_min_frames)
    episodes.sort(key=lambda e: e.t_start)
    return episodes


def _close_run(
    episodes: list[OcclusionEpisode],
    run: list[tuple],
    a: int,
    b: int,
    occl_min_frames: int,
) -> None:
    """Fold a run of high-containment frames into an episode — or drop a
    run too short to be one — and clear it either way."""
    if len(run) >= occl_min_frames:
        boxes = [o.bbox for pair in run for o in pair]
        widths = sorted(o.width for pair in run for o in pair)
        episodes.append(OcclusionEpisode(
            track_a=a,
            track_b=b,
            t_start=run[0][0].t,
            t_end=run[-1][0].t,
            region=(
                min(x[0] for x in boxes), min(x[1] for x in boxes),
                max(x[2] for x in boxes), max(x[3] for x in boxes),
            ),
            scale=widths[len(widths) // 2],
        ))
    run.clear()


def classify_births(
    observations,
    episodes: list[OcclusionEpisode],
    *,
    birth_window_s: float = DEFAULT_BIRTH_WINDOW_S,
    birth_radius_w: float = DEFAULT_BIRTH_RADIUS_W,
    close_gap_s: float = DEFAULT_CLOSE_GAP_S,
) -> tuple[list[int], list[int]]:
    """Which tracks were born inside or just after an occlusion.

    A *mid* birth is a track first observed while an episode is open,
    inside its region, when at least one participant of that episode
    predates it — the "at least one older" clause is what keeps two
    people entering the frame already overlapped from both counting as
    phantoms of each other. A *post* birth is a track first observed
    within `birth_window_s` of an episode *dissolving* and within
    `birth_radius_w` of its region, measured in the episode's own box
    widths. Dissolution is `t_end + close_gap_s`, not `t_end`: an
    episode's last overlapped frame is when the hidden subject stopped
    being detected at all, and they stay invisible for exactly the
    stretch the close gap covers — measuring the window from `t_end`
    would expire it while the subject is still hidden. Each track counts
    once; mid wins.
    """
    first: dict[int, object] = {}
    for o in observations:
        cur = first.get(o.track_id)
        if cur is None or o.t < cur.t:
            first[o.track_id] = o

    mid: list[int] = []
    post: list[int] = []
    for track_id, born in sorted(first.items(), key=lambda kv: kv[1].t):
        is_mid = any(
            e.t_start <= born.t <= e.t_end
            and any(
                p in first and first[p].t < born.t
                for p in (e.track_a, e.track_b)
            )
            and distance_to_region(born.centre, e.region) == 0.0
            for e in episodes
        )
        if is_mid:
            mid.append(track_id)
            continue
        if any(
            e.t_end < born.t <= e.t_end + close_gap_s + birth_window_s
            and e.scale > 0
            and distance_to_region(born.centre, e.region) / e.scale
            <= birth_radius_w
            for e in episodes
        ):
            post.append(track_id)
    return mid, post


def _best_similarity(
    queries: list[list[float]], gallery: list[list[float]]
) -> float | None:
    """Highest cosine similarity between any query and any gallery
    vector; None when either side is empty. Plain arithmetic — the
    vectors are short and few, and this module stays dependency-free."""
    best: float | None = None
    for q in queries:
        qn = sum(x * x for x in q) ** 0.5
        if qn == 0:
            continue
        for g in gallery:
            gn = sum(x * x for x in g) ** 0.5
            if gn == 0:
                continue
            sim = sum(a * b for a, b in zip(q, g)) / (qn * gn)
            if best is None or sim > best:
                best = sim
    return best


def swap_evidence(
    *,
    post_a: list[list[float]],
    pre_a: list[list[float]],
    post_b: list[list[float]],
    pre_b: list[list[float]],
    min_margin: float,
) -> dict | None:
    """Did the tracker swap the two subjects across an occlusion?

    A swap leaves no track dark and mints no track — both bridge and
    birth metrics are blind to it. What it does leave is appearance
    evidence: a track's post-episode crops resembling the *other*
    event's pre-episode crops more than its own, by at least
    `min_margin` (CLD-41's rule: a claim must beat its runner-up, not
    merely clear a bar). Either side crossing is enough to suspect —
    when one subject leaves during the overlap, only one track survives
    to cross — and the returned evidence carries all four scores so the
    verdict is inspectable. Returns None when nothing crossed or when
    a side has no usable frames (no evidence is not evidence of no
    swap — the caller simply cannot judge)."""
    a_own = _best_similarity(post_a, pre_a)
    a_other = _best_similarity(post_a, pre_b)
    b_own = _best_similarity(post_b, pre_b)
    b_other = _best_similarity(post_b, pre_a)
    crossed = []
    if a_own is not None and a_other is not None and a_other - a_own >= min_margin:
        crossed.append("a")
    if b_own is not None and b_other is not None and b_other - b_own >= min_margin:
        crossed.append("b")
    if not crossed:
        return None
    return {
        "crossed": crossed,
        "a_own": a_own, "a_other": a_other,
        "b_own": b_own, "b_other": b_other,
        "min_margin": min_margin,
    }


# -- streaming: the same questions, answered causally ---------------------


class OcclusionMonitor:
    """Per-camera occlusion state over live tracker output.

    Fed once per sampled frame with that frame's detections; returns one
    annotation per detection:

        {"occluded_with": [track_id, ...],   # inside an open episode
         "suspect_birth": None | {"candidates": [track_id, ...],
                                   "reason": "mid" | "post"}}

    Causality costs two things the offline fold does not pay. An episode
    only *confirms* after `occl_min_frames` high-containment frames, so
    a track born on the very frame an overlap begins is flagged on the
    confirmation frame, not its first — the frames in between are the
    ordinary CLD-41 gates' problem (nothing mints off one frame), and
    the suspicion, which is what the stitcher and swap check consume,
    still lands. And the per-episode width sample is capped (bounded
    memory), so on an episode longer than ~5 s the post-birth radius
    unit is the median of its first frames rather than of all of them.

    State is dicts of numbers only, and per camera: track ids restart
    whenever a tracker is rebuilt, which is safe here because everything
    is keyed by recency — a stale id is pruned on `memory_s` long before
    its number can be reissued mid-scene.
    """

    def __init__(
        self,
        *,
        occl_containment: float = DEFAULT_CONTAINMENT,
        occl_min_frames: int = DEFAULT_MIN_FRAMES,
        close_gap_s: float = DEFAULT_CLOSE_GAP_S,
        birth_window_s: float = DEFAULT_BIRTH_WINDOW_S,
        birth_radius_w: float = DEFAULT_BIRTH_RADIUS_W,
        memory_s: float = 600.0,
    ) -> None:
        self.occl_containment = occl_containment
        self.occl_min_frames = occl_min_frames
        self.close_gap_s = close_gap_s
        self.birth_window_s = birth_window_s
        self.birth_radius_w = birth_radius_w
        self.memory_s = memory_s
        #: track_id -> t of first/most recent observation
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        #: track_id -> the caller's class-group key. Only tracks in the
        #: same group can occlude each other here: containment cannot
        #: see depth, so a person whose box sits inside a parked car's
        #: box would otherwise read as "occluded" on every frame of
        #: every visit — permanently gating their learning — and a
        #: cross-class swap is impossible anyway (events never change
        #: class group). None pairs with None, which keeps class-less
        #: callers (tests, the offline harness) working.
        self._group: dict[int, str | None] = {}
        #: (a, b) with a < b -> pair state:
        #: {"high": consecutive high frames, "low": consecutive low
        #:  frames, "open": bool, "t_start": first high t of the run,
        #:  "t_high": last high t, "region": [x1,y1,x2,y2],
        #:  "widths": bounded sample of participant widths}
        self._pairs: dict[tuple[int, int], dict] = {}
        #: recently closed episodes:
        #: {"t_start", "t_end", "region", "scale", "tracks": [a, b]}
        self._closed: list[dict] = []
        #: closures not yet handed to the caller — the swap check runs
        #: once per closed episode, when it closes (`pop_closed`).
        self._fresh_closures: list[dict] = []
        #: track_id -> suspicion, kept so every later frame of a phantom
        #: reports it (the stitcher acts when the first usable embedding
        #: arrives, which is rarely the birth frame).
        self._suspect: dict[int, dict] = {}
        self._last_t: float | None = None

    # -- the one entry point ----------------------------------------------

    def feed(self, t: float, detections: list[dict]) -> list[dict]:
        """Advance one frame; annotate each detection.

        `detections` need `track_id` (int or None) and `bbox`
        ([x1, y1, x2, y2]); entries without a track id get empty
        annotations. Frames must arrive in time order per camera —
        which ingest guarantees, one monitor per camera.
        """
        if self._last_t is not None and t < self._last_t - 1.0:
            # Time went backwards by more than jitter: a different clip
            # segment is being replayed (out-of-order backfill). Pair
            # streaks and closure windows are meaningless across that
            # seam, so the transient episode state resets rather than
            # bridging two unrelated stretches of footage; per-track
            # bookkeeping stays, pruned by its own horizon.
            self._pairs.clear()
            self._closed.clear()
            self._fresh_closures.clear()
        self._last_t = t
        self._prune(t)
        tracked = {}
        for d in detections:
            if d.get("track_id") is None:
                continue
            tid = int(d["track_id"])
            tracked[tid] = tuple(map(float, d["bbox"]))
            self._group[tid] = d.get("group")
        births = [tid for tid in tracked if tid not in self._first_seen]
        for tid in tracked:
            self._first_seen.setdefault(tid, t)
            self._last_seen[tid] = t

        self._advance_pairs(t, tracked)
        self._close_absent(t)

        for tid in births:
            self._classify_birth(tid, t, tracked[tid])

        open_by_track: dict[int, list[int]] = {}
        for (a, b), state in self._pairs.items():
            if state["open"]:
                open_by_track.setdefault(a, []).append(b)
                open_by_track.setdefault(b, []).append(a)

        out = []
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                out.append({"occluded_with": [], "suspect_birth": None})
                continue
            tid = int(tid)
            out.append({
                "occluded_with": sorted(open_by_track.get(tid, [])),
                "suspect_birth": self._suspect.get(tid),
            })
        return out

    # -- internals ---------------------------------------------------------

    def _advance_pairs(self, t: float, tracked: dict[int, tuple]) -> None:
        ids = sorted(tracked)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self._group.get(a) != self._group.get(b):
                    continue
                c = containment(tracked[a], tracked[b])
                state = self._pairs.get((a, b))
                if c >= self.occl_containment:
                    if state is None:
                        state = self._pairs[(a, b)] = {
                            "high": 0, "low": 0, "open": False,
                            "t_start": t, "t_high": t,
                            "region": [*tracked[a]], "widths": [],
                        }
                    if state["high"] == 0:
                        state["t_start"] = t
                        state["region"] = [*tracked[a]]
                        state["widths"] = []
                    state["high"] += 1
                    state["low"] = 0
                    state["t_high"] = t
                    r = state["region"]
                    for box in (tracked[a], tracked[b]):
                        r[0] = min(r[0], box[0])
                        r[1] = min(r[1], box[1])
                        r[2] = max(r[2], box[2])
                        r[3] = max(r[3], box[3])
                        if len(state["widths"]) < 50:
                            state["widths"].append(box[2] - box[0])
                    if not state["open"] and state["high"] >= self.occl_min_frames:
                        state["open"] = True
                        # A participant whose whole life fits inside this
                        # run was born occluded: the overlap is why it
                        # exists, and the older participant is the
                        # subject it is probably a sliver of.
                        for young, old in ((a, b), (b, a)):
                            if (
                                self._first_seen[young] >= state["t_start"]
                                and self._first_seen[old] < state["t_start"]
                            ):
                                self._suspect.setdefault(young, {
                                    "candidates": [old], "reason": "mid",
                                })
                elif state is not None:
                    state["low"] += 1
                    if state["low"] >= self.occl_min_frames:
                        self._close_pair((a, b))

    def _close_absent(self, t: float) -> None:
        for key, state in list(self._pairs.items()):
            if t - state["t_high"] > self.close_gap_s:
                self._close_pair(key)

    def _close_pair(self, key: tuple[int, int]) -> None:
        state = self._pairs.pop(key, None)
        if state is None or not state["open"]:
            return
        widths = sorted(state["widths"])
        episode = {
            "t_start": state["t_start"],
            "t_end": state["t_high"],
            "region": list(state["region"]),
            "scale": widths[len(widths) // 2] if widths else 0.0,
            "tracks": list(key),
            "group": self._group.get(key[0]),
        }
        self._closed.append(episode)
        self._fresh_closures.append(dict(episode))

    def pop_closed(self) -> list[dict]:
        """Episodes closed since the last call — the swap check's cue.

        Draining, so each closure is judged exactly once; the caller
        runs its check in the same frame the closure was noticed, which
        is the first moment both sides can have post-episode frames."""
        fresh, self._fresh_closures = self._fresh_closures, []
        return fresh

    def _classify_birth(self, tid: int, t: float, bbox: tuple) -> None:
        centre = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        # Mid, immediate form: born with its box already contained in an
        # *older* track's box. No episode needs to have confirmed — the
        # containment at birth is itself the signal, and the verdict
        # cannot wait, because the stitch decision happens on this very
        # frame (a phantom's box overlaps the occluder, so the IoU
        # stitch would hand it the occluder's event before the episode's
        # min_frames are up). A false suspect merely costs one skipped
        # IoU stitch; the identity-aware merge still folds a genuine
        # fragment. Two tracks born together stay clean: neither is
        # older, so neither can be the other's phantom.
        for (a, b), state in self._pairs.items():
            if tid not in (a, b) or state["high"] < 1:
                continue
            other = b if tid == a else a
            if self._first_seen.get(other, t) < t:
                # The sliver sits in `other`'s box but most likely
                # belongs to whoever `other` is currently hiding — name
                # both sides of any open episode `other` is in.
                candidates = {other}
                for (a2, b2), st2 in self._pairs.items():
                    if st2["open"] and other in (a2, b2):
                        candidates.update((a2, b2))
                candidates.discard(tid)
                self._suspect[tid] = {
                    "candidates": sorted(candidates), "reason": "mid",
                }
                return
        # Mid, region form: born inside an already-open episode between
        # two older tracks (the sliver that gets its own box beside both
        # participants'). Same-group only — a dog trotting through two
        # overlapped people is its own subject, not their phantom.
        for (a, b), state in self._pairs.items():
            if not state["open"] or tid in (a, b):
                continue
            if self._group.get(tid) != self._group.get(a):
                continue
            if distance_to_region(centre, tuple(state["region"])) == 0.0:
                self._suspect[tid] = {
                    "candidates": sorted((a, b)), "reason": "mid",
                }
                return
        # Post: born shortly after an episode's overlap ended, near its
        # region — the window runs from t_end + close_gap_s because the
        # hidden subject is undetected for exactly that stretch (see
        # classify_births, which uses the same clock). Two sources feed
        # this: episodes already closed, and episodes still nominally
        # open whose last overlapped frame is in the past — the pair
        # does not close until the co-presence gap exceeds close_gap_s,
        # and a subject re-emerging *during* that gap (the designed
        # case) would otherwise be born into a blind spot: too far from
        # the frozen region for the mid rule, invisible to the closed
        # list. Suspicion is assigned only at birth, so a miss here
        # would be permanent.
        windows = [
            (ep["t_end"], ep["region"], ep["scale"], ep["tracks"], ep.get("group"))
            for ep in self._closed
        ]
        for pair, state in self._pairs.items():
            if state["open"] and state["t_high"] < t:
                widths = sorted(state["widths"])
                windows.append((
                    state["t_high"], state["region"],
                    widths[len(widths) // 2] if widths else 0.0,
                    list(pair), self._group.get(pair[0]),
                ))
        for t_end, region, scale, tracks, group in windows:
            if self._group.get(tid) != group:
                continue
            window_end = t_end + self.close_gap_s + self.birth_window_s
            if not (t_end < t <= window_end):
                continue
            if scale <= 0:
                continue
            if (
                distance_to_region(centre, tuple(region)) / scale
                <= self.birth_radius_w
            ):
                self._suspect[tid] = {
                    "candidates": sorted(tracks), "reason": "post",
                }
                return

    def _prune(self, t: float) -> None:
        horizon = t - self.memory_s
        for tid, last in list(self._last_seen.items()):
            if last < horizon:
                self._last_seen.pop(tid, None)
                self._first_seen.pop(tid, None)
                self._suspect.pop(tid, None)
                self._group.pop(tid, None)
        self._closed = [
            ep for ep in self._closed
            if ep["t_end"] >= t - self.birth_window_s - self.close_gap_s
        ]
