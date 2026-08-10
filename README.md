# Siteloom

A modular video & audio intelligence platform for properties with existing,
mismatched camera hardware. Siteloom ingests live streams (UniFi Protect,
generic RTSP) and media archives, detects people/vehicles/animals, identifies
them over time (face ID, vehicle re-ID, plate OCR), tracks sustained noise,
and correlates events against guest bookings — all runnable on a single
laptop today, with the same codebase designed to scale to a distributed
fleet of edge gateways later.

Full product context lives in [`video-processing-platform-prd.md`](video-processing-platform-prd.md).

## How it works

```
[Camera Adapters] -> [Detection (YOLO first-pass)] -> [Identity (face / re-ID / plate)] -> [Stores]
  unifi | rtsp | file        person, vehicle, ...          per-class algorithms          SQLite + Qdrant
                                      |                                                       |
                                 [Job Execution Backend]                                 [Web UI]
                                  local (PoC) | celery/ray (planned)              events, identities, noise
```

Design rules that keep the pieces swappable:

- **Modules never touch a backend.** Detection, identity, and audio modules
  are plain classes with `process(job) -> result`. A Ray-shaped
  `JobDispatcher` (`submit` / `submit_and_wait` / `submit_batch`) wraps
  whichever backend is configured — synchronous `LocalBackend` today,
  Celery/Ray later, with zero application-code changes.
- **First-pass filter.** Expensive recognition only runs on crops where YOLO
  already found something relevant — never on raw frames.
- **Compute/state split.** The identity module only *computes* embeddings
  (serializable, edge-friendly); the central resolver matches them against
  the vector store and owns identity records. Raw video never needs to leave
  the site — only metadata, crops, and embeddings.
- **Config-driven cameras.** Zones, masks, sample rate, and which modules run
  are YAML per camera, not code.

## Identification

Face recognition is a far more mature field than generic person or vehicle
re-identification, so each class gets its own algorithm and threshold:

| Identifier | Algorithm | Notes |
|---|---|---|
| `face` | YuNet detect → align → SFace embed (OpenCV, 128-d) | cosine threshold ≈ 0.36; swappable for InsightFace/ArcFace |
| `person` | ResNet-18 appearance embedding (512-d) | weaker signal; threshold ≈ 0.80+ |
| `vehicle` | ResNet-18 appearance + optional plate OCR | plate and visual signature share one identity record — a match by either wins, and a visual match learns its plate retroactively |
| *any new class* | generic appearance embedding, added dynamically | add the class to `detection.classes` and an identifier + vector collection appear automatically |

Embeddings live in a **local vector database** (Qdrant embedded mode — a
directory, no server); labels, plates, and sighting stats live in **SQLite**.
Identities start in an "unknown" bucket and are named after the fact in the
web UI (label-and-learn) — no pre-enrollment required.

## Media library, labeling, and training

Siteloom also indexes local directories of photos and short videos, sharing
the identity store with live cameras — a face enrolled from a photo archive
is recognized on a camera immediately.

**Indexing is two-phase and resumable**, so an archive too large to process
in one sitting can be done in chunks:

```bash
siteloom library add ~/Pictures/archive     # register a source
siteloom library scan                       # cheap: register files as pending
siteloom library index --limit 200          # expensive: detect + identify 200
siteloom library index --all                # ...come back and finish later
siteloom library status
```

**Labeling** lives at `/library/<id>`: draw and correct bounding boxes on a
canvas, set class and custom sub-class, assign identity, add whole-image
tags, and verify or reject each box. Keyboard-driven (`V` verify, `X`
reject, `N`/`P` next/prev, `Cmd+S` save). Re-indexing never discards human
work — only unverified machine boxes are regenerated.

**Class definition** lives at `/classes`: toggle which detection classes are
tracked, set per-identifier similarity thresholds, and define custom
sub-classes. Changes are written back to your site YAML, so operators
refine classes without editing files.

### Google Photos Takeout

Takeout sidecars (`*.supplemental-metadata.json`) carry a `people` array —
Google Photos' own face grouping. Those names are per *photo*, never per
face box, so the importer assigns them in two passes:

1. **Unambiguous** — one detected face, one person tag. Logically certain;
   auto-verified and used to seed a gallery.
2. **Constrained matching** — for group photos, each face is matched against
   that gallery *restricted to the names tagged on that same photo*.
   Closed-set matching over a handful of candidates is far easier than
   open-set recognition, so this recovers most of an archive. Always
   proposals; always requires review.

```bash
siteloom takeout inspect "~/Takeout/Google Photos"   # dry run, no writes
siteloom takeout import  "~/Takeout/Google Photos"
siteloom takeout status                              # what's named/verified
```

Google exports edited copies as `<name>-edited.jpg` with no sidecar of their
own; they are skipped by default, because importing them alongside their
originals seeds the gallery with near-duplicates and inflates per-person
coverage. Pass `--include-derivatives` to keep them.

Review at `/training`: per-person coverage, confirm/reject/rename in bulk.
**Only what you verify becomes training data** — a model trained on its own
proposals would score well and mean nothing.

### Long-running jobs

Importing a large archive takes a while, so those operations are observable
and interruptible rather than opaque:

- **Progress with ETA** — a live bar with rate and per-phase counters on a
  terminal; periodic log lines when output is redirected, so background runs
  aren't silent either. `--log-file` adds a rotating file log.
- **Ctrl-C is safe** — the current batch finishes, work is committed, the run
  is recorded as interrupted, and the exact resume command is printed (rebuilt
  from the flags you actually passed). Everything is resumable; rerunning skips
  what's done. Files that failed are reported separately from what's pending,
  and `library index --retry-failed` re-queues them.
- **Watch from anywhere** — every batch heartbeats to the database, so a run
  started in one terminal is visible from another and from the browser. A run
  whose process died shows as `stale` with its last position, rather than
  looking healthy forever.
- **Steer from anywhere** — stop a job from a terminal that did not start it,
  and clear up after one that died.

```bash
siteloom jobs list       # recent runs, progress, outcome, resume commands
siteloom jobs watch      # live view of whatever is running
siteloom jobs cancel 12  # graceful stop: finishes the batch, leaves a resume command
siteloom jobs reap       # close out runs whose process is gone
```

The `/jobs` page shows the same thing with live progress bars. SIGTERM and
SIGHUP stop a job as gracefully as Ctrl-C, so a service manager or a closing
terminal costs at most the batch in flight.

### Training

```bash
siteloom train status            # verified samples per person
siteloom train face              # fine-tune the face embedding
siteloom train detector          # train a YOLO face detector
```

`train face` learns a linear projection over SFace features with a
proxy-centroid loss, so *your* people separate better. It is adopted only if
held-out AUC improves on a validation split that can actually be scored —
if there aren't enough samples per person to form same-person validation
pairs, it reports that and keeps the base embeddings rather than pretending
to have improved. Once adopted, the projection is picked up automatically
everywhere embeddings are computed.

`train detector` improves face **detection** on your own imagery (recall on
small, angled, blurred faces). It does nothing for identification — who a
face belongs to remains the embedding pipeline's job.

## Quickstart

Requires Python ≥ 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate               # every command below assumes this
pip install -r requirements-dev.txt     # runtime + test deps, project editable

cp config/site.example.yaml site.yaml   # then edit for your site

siteloom run --config site.yaml     # ingest configured cameras
siteloom serve --config site.yaml   # web UI at http://127.0.0.1:8000
```

Useful commands:

```bash
siteloom cameras                           # list streams adapters can see (UniFi camera ids)
siteloom backfill ~/old-footage            # run a photo/video archive through the same pipeline
siteloom sync-bookings                     # pull guest bookings from the configured iCal feed
pytest                                     # test suite
pip install -r requirements-plates.txt     # optional plate detection + OCR
```

Tests use stub modules and synthetic media, so they need no model weights and
no cameras. What they cannot check by machine — whether a job really survives a
kill, a reboot, or a 26k-item archive — is written up as a runbook in
[docs/testing/resumability-runbook.md](docs/testing/resumability-runbook.md),
with `scripts/make_resume_corpus.py` to build a corpus big enough to interrupt.

For UniFi Protect, put the console host/credentials under `unifi:` in your
site YAML and use each camera's Protect id as its `source`.

## Running it as a service

```bash
siteloom service install --config site.yaml   # writes the unit, starts it
siteloom service status                       # 0 running, 3 stopped, 4 not installed
siteloom service stop | start | restart | logs -f
```

One verb set over launchd (macOS) and systemd (Linux). The unit is generated
from your config, so it carries an absolute program and config path, the right
working directory, a `doctor` preflight, a restart policy with a crashloop
brake, a stop timeout sized to a commit batch, and `SuccessExitStatus=130` so a
`jobs cancel` is not read as a crash and undone. `--unit run|frigate` supervises
the other long-lived commands; `siteloom service print-unit` shows the file
without writing it, and install refuses to clobber a unit it did not write.

Siteloom does not daemonize itself — no `--daemon`, no PID file. The process
runs in the foreground and stops on SIGTERM; the supervisor owns the rest.

```bash
siteloom doctor --config site.yaml   # is this deployment fit to run? exit 1 if not
```

`doctor` checks the database and schema, media dir and free space, the vector
store (including *who is holding it*), model weights, optional plate-OCR deps,
abandoned jobs, integration coherence, and installed service units — each with a
remedy. The server exposes `/healthz` (liveness) and `/readyz` (readiness, 503
when it cannot work), and now heartbeats an `OperationRun` row like every other
long operation, so `siteloom jobs` sees it too.

Which directives are emitted and which are deliberately left out, stop-signal
semantics, and the one-process-per-vector-store rule are in
[docs/operations.md](docs/operations.md).

## Web UI

- **Events** — one card per tracked object visit, with crops, class, camera,
  and time filters; guest-window badges on events during expected arrivals.
- **Library** — indexed local media, filterable by source, status, person
  tag, and "needs review"; click through to the box editor.
- **Identities** — the label-and-learn surface: browse unknowns, name them,
  see every sighting; plates shown on vehicle identities. Merge two
  identities that are the same person, or split a cluster that absorbed two.
- **Classes** — tracked detection classes, identifier thresholds, custom
  sub-class definition.
- **Training** — face proposals with per-person coverage, bulk
  confirm/reject/rename, and a history of training runs.
- **Noise** — sustained loud episodes (dBFS threshold + minimum duration).

## Integrations: Frigate · MQTT · Double Take · CompreFace

Siteloom slots into the self-hosted NVR ecosystem from either side — see
[docs/integrations.md](docs/integrations.md) for the full wiring:

- **`siteloom frigate`** consumes an existing Frigate install's MQTT
  events, fetches the triggering snapshot from Frigate's API, and runs
  face/vehicle recognition on it — the Double Take + CompreFace role,
  with results stored in Siteloom, republished on `siteloom/identity`,
  and fired to webhooks.
- **CompreFace-compatible REST API** (`/api/v1/recognition/...`) on the
  web server, so Double Take or any CompreFace client can use Siteloom
  as its recognizer — recognize, subjects, and face enrollment, with
  `x-api-key` auth.
- **MQTT publishing** from Siteloom's own camera ingest
  (`siteloom/events`, `siteloom/identity`) for Home Assistant-style
  automations, and **webhooks** on `identity.match` /
  `identity.unknown` / `identity.new_plate`.

All of it shares the one identity collection: a face verified from your
photo archive is recognized on a Frigate camera and through the REST API
alike. `siteloom train enroll` sweeps verified review decisions into
that collection (confirmations in the UI enroll automatically).

## Audio & privacy posture

Audio ships with **loud-duration tracking only** — decibel level sustained
past a configurable duration. Sound classification and voice transcription
are deliberately absent; any module touching biometric or conversational
data must be off by default and explicitly enabled per deployment
(see PRD §9 / NFR5 for the legal context).

## Project layout

```
siteloom/
  adapters/    # UniFi Protect, generic RTSP, file/archive
  dispatch/    # JobDispatcher interface + LocalBackend
  modules/     # detection (YOLO+ByteTrack), identity (embeddings), audio
  identity/    # vector store (Qdrant local), embedders, registry, resolver,
               # plates, custom-class k-NN
  library/     # resumable local-directory indexing, Takeout importer
  training/    # verified-sample collection, face fine-tune, YOLO export
  store/       # SQLAlchemy: events, identities, library items, annotations,
               # custom classes, noise, bookings, training runs
  ingest.py    # adapter -> sampler -> dispatcher -> stores
  progress.py  # heartbeated, interruptible, resumable long operations
  health.py    # preflight checks behind `doctor` and /readyz
  web/         # FastAPI + Jinja2 operator UI (events, library, labeling,
               # classes, training review)
  cli.py       # init-db | run | serve | doctor | cameras | backfill
               # sync-bookings | library | takeout | classes | train | jobs
```

## Roadmap

- Hikvision + Dahua adapters (generic RTSP fallback covers them today)
- `CeleryBackend` (durable queue for intermittent machines), then `RayBackend`
- Central dashboard: multi-site identity matching, allow-lists, retention policies
- Live-stream audio, stronger re-ID models (OSNet-class), cross-identifier linking

## License

[AGPL-3.0](LICENSE) — network copyleft: if you run a modified Siteloom as a
service, you must offer its source to your users.
