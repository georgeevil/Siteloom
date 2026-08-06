# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Siteloom: a multi-site video & audio intelligence platform (PoC for Kai Apartments). The spec is `video-processing-platform-prd.md` — read it before making architectural changes; section numbers (§) are referenced throughout the code.

## Commands

- `uv sync --extra dev` — install (uv-managed venv; Python ≥3.12)
- `uv run pytest` — full test suite; single test: `uv run pytest tests/test_ingest.py::test_ingest_end_to_end`
- `uv run siteloom run --config config/site.example.yaml` — ingest configured cameras (add `--max-frames N` for a quick debug run)
- `uv run siteloom serve --config ...` — event-browser web UI on :8000
- `uv run siteloom cameras --config ...` — list streams each adapter can see (used to find UniFi camera ids)
- `uv run siteloom init-db --config ...` — create tables (run/serve also do this implicitly)

Backfill a media archive: `uv run siteloom backfill <path> --config ...`; sync guest bookings: `uv run siteloom sync-bookings`. Plate OCR is optional: `uv sync --extra plates` (without it the vehicle path degrades to visual re-ID with a logged warning).

Library/training sub-apps: `siteloom library add|scan|index|status`, `siteloom takeout inspect|import`, `siteloom classes list|add|rebuild`, `siteloom train status|face|export-detector|detector`. Face models (YuNet/SFace ONNX) auto-download to `~/.cache/siteloom/models`.

## Architecture

Four independently swappable layers (PRD §5); the invariants that keep them swappable matter more than the file layout:

1. **Camera adapters** (`siteloom/adapters/`) — one per brand behind `CameraAdapter` (`connect/list_streams/get_live_stream/get_historical_clip`). All adapters ultimately yield `FrameSource`s wrapping anything OpenCV can decode. `FileAdapter` is both the dev/test path and the seed of the backfill module — backfill must reuse the live pipeline, never fork it (PRD §6.6).
2. **Processing modules** (`siteloom/modules/`) — plain classes with `process(job) -> result` that NEVER import an execution backend. `DetectionModule` (YOLO + ByteTrack) is the first-pass filter: future face/plate/re-ID modules subscribe to its class-routed detections, not raw frames (PRD §6.2). It keeps one YOLO instance per camera because ultralytics stores tracker state on the predictor.
3. **Job execution backend** (`siteloom/dispatch/`) — `JobDispatcher` (submit/submit_and_wait/submit_batch, Ray-shaped API) with `LocalBackend` today; Celery/Ray backends later must slot in with zero application-code changes. `Job` payloads and module results must stay serializable (bytes/dicts/paths — no numpy arrays or live handles) because they will cross process/network boundaries in those backends.
4. **Store** (`siteloom/store/`) — SQLAlchemy on SQLite (Postgres-ready). An `Event` = one track id's visit on one camera; `Detection` rows hang off it. Crops are JPEGs under `media_dir`, path stored on the row.

`siteloom/ingest.py` is the application-layer wiring (adapter → sampler → dispatcher → store); `siteloom/web/` is the FastAPI/Jinja2 operator UI; `siteloom/config.py` holds the pydantic config models — per-camera zones/masks/modules are YAML config, never code (NFR3).

### Identity layer (`siteloom/identity/`)

The compute/state split is deliberate and must be preserved: `modules/identity.py` (IdentityModule) only computes embeddings — serializable, runs at the edge later (NFR2); `identity/resolver.py` (IdentityResolver) owns all identity state — vector search, thresholding, Identity-row creation, plate-first matching — and stays central. Never move DB or vector-store writes into a processing module.

- **Vector DB**: Qdrant in embedded mode (`QdrantClient(path=...)` in `identity/vectors.py`) — a local directory, no server; the same client speaks to remote Qdrant for V1 multi-site. One collection per identifier key, created on demand.
- **Per-class algorithms** (`identity/embedders.py`): face ID uses a dedicated pipeline (YuNet detect → align → SFace 128-d, OpenCV built-ins, models auto-downloaded to `~/.cache/siteloom/models`); everything else uses a shared ResNet-18 appearance embedding (512-d). Each identifier carries its own cosine threshold in config because the similarity distributions differ wildly (face ≈0.36, generic ≈0.8+). The registry (`identity/registry.py`) maps detection class → identifiers and **auto-adds a generic identifier for any unseen class** when `identity.auto_add_classes` is on — adding a class to `detection.classes` is the only step needed to re-identify it.
- **Plates** (`identity/plates.py`): plate OCR and visual re-ID write to the *same* vehicle Identity row (PRD §6.4); a plate match beats visual similarity, and a visual match can learn its plate later.
- **Label-and-learn** (PRD §6.3): identities start unlabeled ("unknown-…" bucket in `/identities`); labeling via the web UI renames all past and future matches — never require pre-enrollment.

### Library, labeling, training (`siteloom/library/`, `siteloom/training/`)

- **Two-phase indexing is the whole point of `library/indexer.py`**: `scan()` is cheap and registers rows as `pending`; `process(limit=N)` is expensive and bounded. Never collapse them — partial/resumable indexing over huge archives depends on the split. Re-indexing deletes only unverified `source="auto"` annotations; human work must always survive.
- **`Annotation` is one table for four jobs** (detection review, identity labeling, custom-class labeling, face training data). `source` records provenance (`auto`/`human`/`import`), `verified` records human sign-off, `rejected` keeps negatives rather than deleting them. Boxes are stored **normalized 0..1** so they survive thumbnailing.
- **Only verified, non-rejected annotations are training data** (`training/dataset.py`). The Takeout importer's proposals live in `proposed_name`/`proposal_basis`, deliberately separate from `identity_id`, so guesses never leak into the identity store or a training export.
- **Takeout two-pass assignment** (`library/takeout.py`): Google's `people` tags are per-photo, not per-face. Pass 1 handles 1-face-1-name (certain, auto-verified, seeds a gallery); pass 2 matches remaining faces against that gallery *restricted to names tagged on the same photo*. Sidecar matching is by the JSON `title` field first, filename heuristics second — Takeout's naming is genuinely inconsistent (truncation, `(1)` counters that migrate into the JSON name, legacy `.json` suffixes).
- **Never adopt a model without a valid evaluation** (`training/face.py`). `EvalMetrics.valid` is False when a split can't produce both same- and different-person pairs; an invalid score is not a zero score. `split_by_person` enforces a 2-sample validation floor per person so same-person pairs exist. Fine-tuning is only adopted on a genuine held-out improvement — evaluating on train can never justify it.
- **Custom sub-classes are k-NN, not a model** (`identity/classes.py`): examples are labeled crops embedded with the existing generic embedder, stored in the `class-examples` collection. Adding an example improves the class immediately; there is no training run. Rebuild after the embedder changes, or stale vectors from an old embedding space silently degrade voting.
- YOLO face-detector training improves **detection only**; identification stays with the embedding pipeline. Say so plainly rather than implying otherwise.

### Audio, backfill, guests

- `modules/audio.py`: loud-duration episodes only (dBFS RMS threshold + min duration + release gap) — classification/transcription intentionally absent (NFR5). `detect_episodes` is a pure function; test changes there.
- Backfill (`siteloom backfill`) builds a synthetic file-adapter CameraConfig and runs `IngestService.run_camera` — it must always reuse the live pipeline (PRD §6.6).
- `guests.py`: iCal → Booking rows; `GuestWindows.contains()` stamps `Event.guest_window` at ingest to suppress unknown-vehicle alarms during arrival windows (the PRD §12 success metric).

## Constraints to preserve

- Primary target is Apple Silicon (`device: mps`); later a heterogeneous fleet — keep model/runtime choices config-driven.
- Zone hit-testing uses the bbox bottom-center point (Frigate convention) — see `_zones_hit` in `modules/detection.py`.
- Modules touching biometric/conversational data must be **off by default** (NFR5); voice transcription stays disabled pending legal review (PRD §9).
- Tests must not require model weights or live cameras: stub modules + synthetic videos (see `tests/conftest.py`); the real-YOLO path is verified via the manual smoke run against `samples/`.
- License is AGPL-3.0 — network copyleft; keep it in mind when vendoring code.
