# Runbook: is face + vehicle ID holding up on the live cameras? (CLD-5)

Scope: the live UniFi path end to end — `siteloom run` against the real
Protect cameras, `siteloom serve` beside it, and the accuracy readout at
`/stats` that turns a night of events into a number.

The claim under test is the one the thresholds encode: that a face match at
**≥0.36** and a generic match at **≥0.80–0.82** cosine are right often
enough to act on, on *these* cameras at *these* angles. Nothing in the
repository establishes that — the defaults came from the embedders'
published distributions, not from Kai Apartments. Three things have to hold:

1. **Precision** — claims the system makes are right; wrong ones are rare
   and cluster near the threshold rather than scattering above it.
2. **Recall** — subjects that should have been identified were, and misses
   are attributable to an identifier rather than being invisible.
3. **Survivability** — the run lasts the night; drops are counted, not
   silent, and a crash at 3am is visible as a crash.

Results go in a dated section at the bottom, as numbered findings, the way
[resumability-runbook.md](resumability-runbook.md) does it. Day-to-day
operation of a running deployment is in [operations.md](../operations.md).

## 0. Setup

### Prerequisites now in place

- `siteloom run` heartbeats an `OperationRun` row (CLD-15), so the soak is
  visible to `siteloom jobs`, `jobs watch` and `/jobs` from another
  terminal, and `kill -9` leaves a stale row rather than a healthy-looking
  one. Counters are per camera: `frames`, `detections`, `matches`,
  `reconnects`, plus `<camera>.<counter>`.
- `/stats` (CLD-17) reports per-identifier claims/confirmed/wrong/misses,
  the plate-vs-visual split, and the similarity distribution against each
  configured threshold. Every rate carries its denominator.
- Verdicts (CLD-16) already exist on the event rail: ✓ / ✗ per claim, plus
  "missed" for a subject the system said nothing about.

### Config

`config/site.kai.yaml` is the committed template — three UniFi cameras at
5 fps. Copy it to `site.yaml` and fill in the Protect password; the file is
gitignored precisely so the secret never lands in the repo.

```bash
cp config/site.kai.yaml site.yaml
$EDITOR site.yaml            # unifi.password
.venv/bin/siteloom doctor --config site.yaml
```

`doctor` first, always: it reports the two things that most often waste a
soak — a vector store already held by another process, and plate OCR
requested by config but not installed.

Two settings worth deciding before starting rather than after:

- **`sample_fps: 5.0`** — keep it. Walking-pace subjects below ~5 fps break
  the tracker (CLD-5/CLD-40 fragmentation), and the soak is meant to test
  identity, not to re-discover that.
- **`audio.enabled: false`** — leave it. Audio cannot reach the audio
  module on a live stream at all: `Frame` carries only an image, OpenCV's
  FFmpeg backend discards audio streams, and ingest only calls
  `_process_audio` for `adapter == "file"`. Flipping the flag changes
  nothing. See scenario 5.

### Before starting

- **Stop any other `siteloom` process on the same `vector_db_path`.**
  Embedded Qdrant is one client per path per machine; `serve` and `run`
  share the store through `get_shared_store`, but a stray CLI does not.
- Note the free disk. Crops accumulate under `media_dir/<camera>/<date>/`
  and a three-camera night at 5 fps is the first real measurement of that
  growth rate — scenario 4 asks for the number.

## 1. Scenario A — start the soak

```bash
# Terminal 1 — ingest, logging to a file so the night is reviewable.
.venv/bin/siteloom run --config site.yaml --log-file soak.log

# Terminal 2 — the console.
.venv/bin/siteloom serve --config site.yaml
```

Within a minute, confirm from a **third** terminal that the run is visible
— this is the CLD-15 promise, and if it is broken the rest of the night is
unobserved:

```bash
.venv/bin/siteloom jobs list --config site.yaml
```

Expect one `run` row, status `running`, with a fresh heartbeat and non-zero
`frames`. `jobs watch` follows it live.

Let it run **≥24 h**, ideally covering a guest arrival window — unknown-
vehicle suppression during arrivals is the PRD §12 success metric, and it
is the one result that cannot be reconstructed later.

## 2. Scenario B — review every event, file the verdicts

This is the part that produces data. On `/` (Events):

- Work the **needs review** and **unmatched** chips.
- For each identity claim: **✓** if the name is right, **✗** if it is
  wrong. Do not skip the obvious ones — reviewing only the suspicious
  claims is exactly the sampling bias `/stats` reports coverage to expose.
- For a subject the system named nothing on: mark **missed**, and pick the
  identifier that should have caught it (face vs person vs vehicle). A
  miss without an identifier is not attributable, so it cannot become
  recall.
- Label a handful of recurring unknowns. Labelling is what makes the next
  night better, and it exercises label-and-learn on live data.

**Clearing an event needs no verdict** — that is deliberate, since most
events have nothing to judge. `/stats` counts how many events were cleared
without one, so the blind spot stays visible; just don't clear the queue
wholesale to get to zero.

## 3. Scenario C — read the accuracy

Open `/stats` (24h window). Record, per identifier:

- claims, confirmed, wrong, **and the reviewed denominator** — a wrong rate
  over 4 of 300 claims is not a measurement.
- misses, and whether recall differs per camera (a bad angle shows up here
  before anywhere else).
- vehicles: plate vs visual split, and how many identities learned a plate
  later — PRD §6.4 working as designed, or not working at all.
- the similarity histogram per identifier: is the wrong-verdict mass
  clustered just above the cutoff (a threshold problem, fixable) or spread
  across the range (an embedder problem, CLD-44)?

The page states the trade-off directly: how many known-wrong matches
raising the cutoff would remove, and how many confirmed ones sit close
enough above it to be at risk.

## 4. Scenario D — the operational facts

From `siteloom jobs list` and `soak.log`:

- `reconnects` per camera. A camera reconnecting hourly is a finding.
- sustained frame rate on `mps` — `frames` ÷ elapsed, per camera, against
  the configured 5 fps. Falling short means the detector, not the network.
- `du -sh media/` before and after: crop growth per night, which is what a
  retention policy will have to be sized against.
- Did the run survive? A `run` row still `running` with a cold heartbeat
  means it died; `siteloom jobs reap` closes it out as `abandoned`.

## 5. Scenario E — what this runbook deliberately cannot test

Two of the M0/M1 spikes cannot be executed as written. Recording why, so
they are not attempted and quietly marked done:

- **Noise thresholds (CLD-7)** — no audio reaches the module on a live
  stream, as above. The only live-camera-derived audio path is
  `backfill-unifi`, which downloads MP4s and does run `_process_audio`. So
  the honest version of this spike is: export a few evening windows with
  `backfill-unifi`, then check `/noise` against what those clips actually
  sound like. That is a rescope, not a run of this runbook.
- **Motorcycle plate OCR (CLD-9)** — *was* blocked here and is not any
  more (CLD-85). Nothing used to record what the OCR read: `read()`
  returned a bare string or None, the plate detector's confidence picked a
  box and was discarded, and a hard `len(text) >= 4` filter dropped short
  reads without trace — so counting hits by eye produced a number that
  could not be reproduced or compared. Every attempt is now a `PlateRead`
  row (raw text before normalization, both confidences, the plate
  sub-crop, and the reason a read was rejected), and `/plates` lists them
  filtered by class with a confirm/reject per row. The spike is now: run
  the soak, open `/plates?class=motorcycle`, judge twenty rows. If the
  four-character floor is the problem, move
  `identity.identifiers.vehicle.plate_min_chars` and re-read the same
  table — the rejected reads kept their text, so nothing is re-run.

## 6. Scenario F — the archive, in parallel

Independent of the cameras, and the other half of "does this work on real
data" (CLD-6). The Takeout archive is already registered — 26,035 items,
all still `pending`, because the importer registers and detects faces but
never marks items `indexed`.

```bash
.venv/bin/siteloom library status --config archive.yaml
.venv/bin/siteloom library index --config archive.yaml --all --log-file index.log
```

Multi-hour and resumable: Ctrl-C prints a resume command that works. Run it
against a **copy** of `archive.db` if you want the option of throwing the
result away. Afterwards, `siteloom train status` says whether any person
has cleared the 5-verified-sample floor that face fine-tuning (CLD-10)
needs — that is the gate on that spike, and it is a labelling-volume
problem, not a code problem.

## Results

_To be filled in after the first pass, as numbered findings F1…Fn, each
marked `(fixed)`, open, or `(split out: CLD-nn)` — the convention from
[resumability-runbook.md](resumability-runbook.md#results-2026-08-06)._

The outcome CLD-5 is waiting on is a written verdict: precision and recall
per identifier with their denominators, whether the 0.36 / 0.8+ thresholds
hold on these cameras, and a ranked list of what to fix.
