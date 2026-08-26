# Is a tracker change actually an improvement? (CLD-98)

Tracker settings trade two failures against each other:

- **ID switches** — one track absorbs two people. Every identity claim on
  that event is then about the wrong person. This is what event 1392 did,
  producing fifteen wrong claims off a single track (CLD-97).
- **Fragmentation** — one person becomes many tracks. Cheaper, since the
  stitcher exists to undo it, but it inflates event counts and starves
  each fragment of evidence.

**Any knob can zero one of these by wrecking the other.** `match_thresh:
0.5` removes every switch — by producing four times the tracks. So a
configuration is only better if both numbers are on the table, and that
is what this harness refuses to let you skip.

The corpus lives in [`config/track_corpus.yaml`](../../config/track_corpus.yaml)
and the driver is [`scripts/track_ab.py`](../../scripts/track_ab.py).

## What is and isn't in the repository

Committed: **which minute of which camera, and what failure it is a case
of.** Not committed: the footage. These are minutes of real video from a
residential property and each is ~16 MB. `fetch` pulls them from UniFi
Protect into `~/.cache/siteloom/track-corpus/`, skipping what is already
there, so the corpus reconstitutes on any machine with NVR access.

This needs model weights and (to fetch) a reachable NVR, which is why it
is a script rather than a test. The arithmetic it rests on is in
`siteloom/track_eval.py` and *is* unit-tested, against synthetic
timelines with known answers.

## Setup

```bash
.venv/bin/python scripts/track_ab.py fetch
```

Reads camera ids and NVR credentials from `site.yaml`. Idempotent — safe
to re-run after adding a clip.

## Comparing configurations

```bash
.venv/bin/python scripts/track_ab.py run
.venv/bin/python scripts/track_ab.py run --only botsort-reid
.venv/bin/python scripts/track_ab.py run --json /tmp/before.json
```

Output as of the current shipped config (2026-08-25, BoT-SORT+ReID) —
verbatim from a `run`, trimmed to the historical variants:

```
event-1392-two-people  (backyard-puerta)
  shipped              tracks   8  det  92%  step-IoU 0.86  bridges  2 (0 implausible)  worst   3.2s/0.3w  occl  2  births 1mid/1post
  bytetrack-2s         tracks   9  det  92%  step-IoU 0.88  bridges  1 (0 implausible)  worst   3.0s/0.3w  occl  8  births 3mid/2post
  upstream-defaults    tracks   8  det  92%  step-IoU 0.88  bridges  3 (1 implausible)  worst   4.8s/1.2w  occl  8  births 3mid/2post
  loose-match          tracks  25  det  91%  step-IoU 0.88  bridges  1 (0 implausible)  worst   3.0s/0.2w  occl  6  births 3mid/5post
    vs shipped — bytetrack-2s: worse  (implausible 0→0, births 2→5, tracks 8→9)
    vs shipped — upstream-defaults: worse  (implausible 0→1, births 2→5, tracks 8→8)
    vs shipped — loose-match: rejected: bought it with fragmentation  (implausible 0→0, births 2→8, tracks 8→25)
backyard-puerta-occlusion-two-people  (backyard-puerta)
  shipped              tracks   2  det  84%  step-IoU 0.88  bridges  0 (0 implausible)  worst           —  occl  0  births 0mid/0post
  bytetrack-2s         tracks   3  det  85%  step-IoU 0.87  bridges  0 (0 implausible)  worst           —  occl  0  births 0mid/0post
    vs shipped — bytetrack-2s: rejected: bought it with fragmentation  (implausible 0→0, births 0→0, tracks 2→3)
```

(That last verdict is the 1.25× fragmentation rule doing its job on
small numbers — 3 tracks against 2 is the old config failing to fold
B's edge fragment back, not a new pathology.)

## Reading the columns

**tracks** — distinct ids minted. Rising sharply is fragmentation.

**bridges** — re-acquisitions after a gap longer than `bridge_gap_s`. A
track that went dark for seconds and resumed elsewhere was reconnected by
a motion prediction, not by evidence. Not every bridge is a switch (people
do walk behind trees), so the count is an upper bound.

**implausible** — bridges where the box moved **a full box width or more**.
That is the point where old and new boxes cannot overlap at all, so
nothing but the prediction connects them. This is the number that matters.

Distance is reported in box widths (`w`), never pixels. 165 px is damning
for a 100 px-wide subject and unremarkable for a 400 px one, and each
track is judged against *its own* median width so a distant subject is not
measured against a close one.

**step-IoU** — overlap between genuinely consecutive frames. This answers
a different question: *is the sample rate fast enough for this subject?*
0.86 means comfortably yes. If this collapses toward 0.2, raise
`sample_fps` — that is the CLD-5 fragmentation mode, and no tracker
setting fixes it.

**worst** — ranked by distance, not duration. A 30-second gap the subject
barely moved across is a successful re-acquisition; a 3-second one they
crossed the frame in is not.

**occl** — occlusion episodes ("crossings"): stretches where two
co-present tracks held **containment** (intersection over the *smaller*
box's area) ≥ 0.5 for at least 2 sampled frames. Containment, not IoU: a
hidden person's partial box is small relative to the occluder's, so
their IoU stays low exactly when the occlusion is total. Reported so
that zero births over zero crossings is visibly "nothing happened"
rather than "nothing went wrong".

**births** — the occlusion failures bridges cannot see, because when one
person walks behind another *neither track goes dark*:

- **mid** — a track first observed inside an open episode's region while
  at least one participant of that episode predates it. The sliver of a
  hidden person's arm minting a fresh id.
- **post** — a track first observed within 3 s of an episode
  *dissolving* (its last overlapped frame plus the close gap — the
  stretch the hidden subject is not detected at all) and within one box
  width of where it happened. The hidden person stepping out and being
  greeted as a stranger.

Both count on the same axis as implausible bridges in the verdict: a
subject acquiring an identity it should not have. A config that trades
one for the other has not improved anything. What the births cannot see
is a **swap** — the tracker putting A's id on B when both boxes reappear.
No track is born, so no clip-level metric fires; catching swaps needs
appearance evidence and lives in the ingest-side occlusion layer, not
here.

## The regression gate

```bash
.venv/bin/python scripts/track_ab.py check   # exit 1 if any clip regresses
```

Runs only the shipped configuration and compares against the `expect`
block recorded beside each clip (`max_tracks`, `max_implausible_bridges`,
`max_mid_occlusion_births`, `max_post_occlusion_births` — any subset). Run it after any change to
`TRACKER_DEFAULTS`, `detection.model`, `detection.confidence`, or
`sample_fps` — all of which move these numbers.

Not in CI: it needs weights and cached footage. It is a habit, not a hook.

## Adding a case

This is the point of the whole thing. When an event turns out to hold two
people, or one person turns into six events:

1. Note the camera and a padded time window around it.
2. Add a `clips:` entry with a `case:` describing what went wrong — in
   plain terms, so a reader in six months knows what the clip is *for*.
3. `fetch`, then `run` to see where the shipped config lands.
4. Add an `expect:` block with a little headroom. It should fail on a
   regression, not on a model that detects one extra frame.

A clip with no `case` is a curiosity; the description is what makes it a
regression test.

## Findings so far

| date | finding |
|---|---|
| 2026-08-09 | `track_buffer: 30` (ultralytics' default) is six seconds of coasting at 5 fps — it merged two people into event 1392. Cut to 10. (CLD-97) |
| 2026-08-09 | `match_thresh: 0.5` zeroes bridges by fragmenting 7 tracks into 18. Rejected. |
| 2026-08-09 | BoT-SORT **with ReID** scores identically to plain ByteTrack here. The appearance-aware tracker that looks obviously right bought nothing. Worth re-running when ultralytics changes its ReID backend — "no benefit" is a measurement, not a permanent fact. |
| 2026-08-09 | `yolo11s` costs 40 ms/frame warm against nano's 39 on mps at this resolution, and halves track churn. Effectively free. |
| 2026-08-25 | The occlusion metrics landed and immediately showed 1392 had occlusion phantoms all along: 3 mid + 2 post births under plain ByteTrack, invisible to the bridge count. |
| 2026-08-25 | The occlusion clip itself is clean under every config at 5 fps — the hidden person is simply not *detected* until they emerge. Its `expect` pins the zero-birth budget for the day a model does see the sliver. |
| 2026-08-25 | **Shipped moved to BoT-SORT + ReID (`model: auto`) + 4 s derived buffer + `new_track_thresh: 0.5`.** Cut 1392's births 5→2 (strict birth kills sliver mints, appearance-verified re-acquisition kills post-emergence strangers) and folded the occlusion clip's edge fragment back into its subject (3 tracks → 2 = two people). "ReID buys nothing" from 2026-08-09 was measured on a *bridge* case; on occlusion cases it is the whole fix. |
| 2026-08-25 | `track_buffer` is now derived: `track_buffer_s` (seconds) × `sample_fps`. The fixed frame count silently meant 5 s at 2 fps and 2 s at 5 fps. 4 s is safe only alongside ReID — buffer length and appearance verification are one decision (CLD-96). |
| 2026-08-25 | A dedicated ReID encoder (`yolo11n-cls.pt`) and the appearance gate in both directions (0.6 looser, 0.9 stricter — note `appearance_thresh` gates distance at `1 − thresh`, so *lower is looser*) all score identically to shipped. Re-run when the corpus gains an IR camera — detector features are weakest there. |

## A result belongs to a camera, not to the site

Everything above was measured on one minute of `backyard-puerta`: gravel
and foliage, subjects 2–5 m away and roughly 100 px wide, at night under
IR. None of that generalises. A driveway watching cars at 30 m, or a
doorway where subjects fill the frame, will disagree about every number
here — box widths differ by an order of magnitude, so the same pixel jump
is damning on one camera and unremarkable on the next, and IR washes out
exactly the colour the appearance embedder leans on.

So `shipped` winning on this clip means *the shipped config is right for
this camera*. It is evidence about a site only once the corpus covers
several cameras that disagree with each other.

**Detection settings are not per-camera yet** (CLD-99). `sample_fps`,
event rules and identity thresholds already are; `detection.model`,
`.confidence`, `.class_confidence` and `.tracker` are site-wide, so today
the harness can only recommend one setting for all cameras and the best
it can do is tell you which camera is paying for that. Until that lands,
read a per-clip win as "this does not hurt here" rather than "adopt this
everywhere".

The practical consequence for the corpus: **add clips from cameras that
are unlike each other** before drawing a conclusion. Two clips from the
same camera confirm each other; two from different cameras are the first
real test of a setting.

## What this does not measure

- **Whether an identity claim was right.** That is `/stats` and operator
  verdicts. This measures the tracking those claims are built on.
- **True ID switches.** Only bridges, which are their observable
  precursor. Ground truth would need per-frame labelling that no config
  change would survive.
- **Stitching.** `stitch_min_iou` and `stitch_gap_s` act on events, after
  tracking. Event 1392 was never stitched — one track id throughout — so
  the stitcher is out of scope here and still unmeasured.
