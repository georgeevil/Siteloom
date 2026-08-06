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
- **Identities** — the label-and-learn surface: browse unknowns, name them,
  see every sighting; plates shown on vehicle identities.
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
  identity/    # vector store (Qdrant local), embedders, registry, resolver, plates
  store/       # SQLAlchemy models: events, detections, identities, noise, bookings
  ingest.py    # adapter -> sampler -> dispatcher -> stores
  web/         # FastAPI + Jinja2 operator UI
  cli.py       # init-db | run | serve | cameras | backfill | sync-bookings
```

## Roadmap

- Hikvision + Dahua adapters (generic RTSP fallback covers them today)
- `CeleryBackend` (durable queue for intermittent machines), then `RayBackend`
- Central dashboard: multi-site identity matching, allow-lists, retention policies
- Live-stream audio, stronger re-ID models (OSNet-class), cross-identifier linking

## License

[AGPL-3.0](LICENSE) — network copyleft: if you run a modified Siteloom as a
service, you must offer its source to your users.
