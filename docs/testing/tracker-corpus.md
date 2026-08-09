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

Output as of the current shipped config:

```
event-1392-two-people  (backyard-puerta)
  shipped            tracks   7  det  87%  step-IoU 0.86  bridges 1 (0 implausible)  worst 4.0s/0.4w
  upstream-defaults  tracks   7  det  87%  step-IoU 0.86  bridges 2 (1 implausible)  worst 4.8s/1.2w
  loose-match        tracks  18  det  86%  step-IoU 0.87  bridges 1 (0 implausible)  worst 4.0s/0.3w
  botsort-reid       tracks   7  det  87%  step-IoU 0.85  bridges 1 (0 implausible)  worst 4.0s/0.3w
    vs shipped — upstream-defaults: worse
    vs shipped — loose-match: rejected: bought it with fragmentation
    vs shipped — botsort-reid: no change
```

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

## The regression gate

```bash
.venv/bin/python scripts/track_ab.py check   # exit 1 if any clip regresses
```

Runs only the shipped configuration and compares against the `expect`
block recorded beside each clip. Run it after any change to
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

## What this does not measure

- **Whether an identity claim was right.** That is `/stats` and operator
  verdicts. This measures the tracking those claims are built on.
- **True ID switches.** Only bridges, which are their observable
  precursor. Ground truth would need per-frame labelling that no config
  change would survive.
- **Stitching.** `stitch_min_iou` and `stitch_gap_s` act on events, after
  tracking. Event 1392 was never stitched — one track id throughout — so
  the stitcher is out of scope here and still unmeasured.
