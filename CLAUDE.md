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

## Architecture

Four independently swappable layers (PRD §5); the invariants that keep them swappable matter more than the file layout:

1. **Camera adapters** (`siteloom/adapters/`) — one per brand behind `CameraAdapter` (`connect/list_streams/get_live_stream/get_historical_clip`). All adapters ultimately yield `FrameSource`s wrapping anything OpenCV can decode. `FileAdapter` is both the dev/test path and the seed of the backfill module — backfill must reuse the live pipeline, never fork it (PRD §6.6).
2. **Processing modules** (`siteloom/modules/`) — plain classes with `process(job) -> result` that NEVER import an execution backend. `DetectionModule` (YOLO + ByteTrack) is the first-pass filter: future face/plate/re-ID modules subscribe to its class-routed detections, not raw frames (PRD §6.2). It keeps one YOLO instance per camera because ultralytics stores tracker state on the predictor.
3. **Job execution backend** (`siteloom/dispatch/`) — `JobDispatcher` (submit/submit_and_wait/submit_batch, Ray-shaped API) with `LocalBackend` today; Celery/Ray backends later must slot in with zero application-code changes. `Job` payloads and module results must stay serializable (bytes/dicts/paths — no numpy arrays or live handles) because they will cross process/network boundaries in those backends.
4. **Store** (`siteloom/store/`) — SQLAlchemy on SQLite (Postgres-ready). An `Event` = one track id's visit on one camera; `Detection` rows hang off it. Crops are JPEGs under `media_dir`, path stored on the row.

`siteloom/ingest.py` is the application-layer wiring (adapter → sampler → dispatcher → store); `siteloom/web/` is the FastAPI/Jinja2 operator UI; `siteloom/config.py` holds the pydantic config models — per-camera zones/masks/modules are YAML config, never code (NFR3).

## Constraints to preserve

- Primary target is Apple Silicon (`device: mps`); later a heterogeneous fleet — keep model/runtime choices config-driven.
- Zone hit-testing uses the bbox bottom-center point (Frigate convention) — see `_zones_hit` in `modules/detection.py`.
- Modules touching biometric/conversational data must be **off by default** (NFR5); voice transcription stays disabled pending legal review (PRD §9).
- Tests must not require model weights or live cameras: stub modules + synthetic videos (see `tests/conftest.py`); the real-YOLO path is verified via the manual smoke run against `samples/`.
- License is AGPL-3.0 — network copyleft; keep it in mind when vendoring code.
