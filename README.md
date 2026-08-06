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
uv run siteloom library add ~/Pictures/archive     # register a source
uv run siteloom library scan                       # cheap: register files as pending
uv run siteloom library index --limit 200          # expensive: detect + identify 200
uv run siteloom library index --all                # ...come back and finish later
uv run siteloom library status
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
uv run siteloom takeout inspect "~/Takeout/Google Photos"   # dry run, no writes
uv run siteloom takeout import  "~/Takeout/Google Photos"
uv run siteloom takeout status                              # what's named/verified
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
  is recorded as interrupted, and the exact resume command is printed.
  Everything is resumable; rerunning skips what's done.
- **Watch from anywhere** — every batch heartbeats to the database, so a run
  started in one terminal is visible from another and from the browser. A run
  whose process died shows as `stale` with its last position, rather than
  looking healthy forever.

```bash
uv run siteloom jobs list     # recent runs, progress, outcome, resume commands
uv run siteloom jobs watch    # live view of whatever is running
```

The `/jobs` page shows the same thing with live progress bars.

### Training

```bash
uv run siteloom train status            # verified samples per person
uv run siteloom train face              # fine-tune the face embedding
uv run siteloom train detector          # train a YOLO face detector
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

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev            # creates ./.venv and installs everything
cp config/site.example.yaml site.yaml   # then edit for your site

uv run siteloom run --config site.yaml     # ingest configured cameras
uv run siteloom serve --config site.yaml   # web UI at http://127.0.0.1:8000
```

Useful commands:

```bash
uv run siteloom cameras                    # list streams adapters can see (UniFi camera ids)
uv run siteloom backfill ~/old-footage     # run a photo/video archive through the same pipeline
uv run siteloom sync-bookings              # pull guest bookings from the configured iCal feed
uv run pytest                              # test suite
uv sync --extra plates                     # optional plate detection + OCR
```

For UniFi Protect, put the console host/credentials under `unifi:` in your
site YAML and use each camera's Protect id as its `source`.

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
  web/         # FastAPI + Jinja2 operator UI (events, library, labeling,
               # classes, training review)
  cli.py       # init-db | run | serve | cameras | backfill | sync-bookings
               # library | takeout | classes | train
```

## Roadmap

- Hikvision + Dahua adapters (generic RTSP fallback covers them today)
- `CeleryBackend` (durable queue for intermittent machines), then `RayBackend`
- Central dashboard: multi-site identity matching, allow-lists, retention policies
- Live-stream audio, stronger re-ID models (OSNet-class), cross-identifier linking

## License

[AGPL-3.0](LICENSE) — network copyleft: if you run a modified Siteloom as a
service, you must offer its source to your users.
