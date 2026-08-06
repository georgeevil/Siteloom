# Product Requirements Document
## Multi-Site Video & Audio Intelligence Platform

**Status:** Draft v0.1 — PoC scoping
**Owner:** George
**Last updated:** 2026-08-05

---

## 1. Summary

A modular platform that ingests video/audio from existing camera systems
(Hikvision, Dahua, UniFi Protect, generic RTSP/ONVIF), detects and identifies
people, vehicles, and audio events over time, and supports both a single-site
deployment (Kai Apartments PoC) and a future multi-site, multi-tenant B2B
product for property managers.

The system must work whether it's running as one process on one laptop, or
as a distributed fleet of edge gateways feeding a central platform — same
codebase, different execution backend.

## 2. Goals

- G1: Replace/augment existing UniFi cameras at Kai Apartments with LPR,
  face ID, vehicle re-ID (for plates that aren't visible), and audio-event
  detection, without requiring new camera hardware.
- G2: Support ingesting and backfilling historical photos/video already on
  disk into the same identity database used for live detection.
- G3: Prove the architecture scales conceptually from "one property" to
  "many small sites reporting to a central platform" without a rewrite.
- G4: Keep the processing framework itself decoupled from *how* jobs run —
  a single laptop running everything synchronously for the PoC, or a
  distributed Celery/Ray fleet later, behind the same interface.

## 3. Non-goals (for the PoC)

- NG1: Not building a shared cross-customer law-enforcement network (Flock's
  model) — out of scope entirely.
- NG2: Not shipping voice-content transcription of passers-by in the PoC —
  legal review required first (see §9, Risks).
- NG3: Not supporting every camera brand on day one — three adapters
  (Hikvision, Dahua, UniFi Protect) is the target, not an open-ended list.
- NG4: Not building multi-tenant billing/auth for the PoC — single-site,
  single-operator only.

## 4. Users

- **Primary (PoC):** George, operating Kai Apartments — needs to know when
  an unrecognized vehicle or person is on the property, correlate that
  against guest check-in data, and get a noise-duration alert for guest
  units.
- **Future (V1):** Small property-management companies running many
  small sites (vacation rentals, boutique hotels) with existing,
  mismatched camera hardware and no in-house IT staff.

## 5. Architecture overview

Four layers, each independently swappable:

```
[Camera Adapter Layer] -> [Detection/Analytics Modules] -> [Job Execution Backend] -> [Storage & Identity Store]
                                                                   |
                                                          [Central Dashboard/API]
                                                          (optional in PoC; required in V1)
```

The key modularity requirement: **the Detection/Analytics Modules never call
a specific execution backend directly.** They're plain Python callables
wrapped by whichever backend is active (see §7). This is what makes
"runs on one laptop" and "runs distributed across 3 Macs and a central
server" the same codebase.

## 6. Functional modules

### 6.1 Camera Adapter Module
- One adapter per brand, implementing a shared interface:
  `connect()`, `list_streams()`, `get_live_stream(id)`,
  `get_historical_clip(id, start, end)`.
- V0.1 targets: Hikvision (ISAPI/RTSP), Dahua (RTSP/SDK), UniFi Protect
  (REST API via `uiprotect`/pyunifiprotect-style client).
- Generic RTSP/ONVIF fallback adapter for anything not explicitly supported
  — lower feature set (stream only, no event API), but keeps the system
  from being blocked on a brand it hasn't been built for yet.
- **From Frigate:** treat each camera as config-driven, not hardcoded —
  a YAML/JSON entry per camera defining zones, motion masks, and which
  detection modules apply to it (e.g. skip face ID on a driveway-only camera).

### 6.2 Detection Module (video)
- Object detection (person, vehicle, bicycle, animal) via YOLO.
- Runs as a **first pass filter** — face/plate/re-ID modules only run on
  regions where the first pass already found something relevant. This is
  the single highest-value idea to borrow from Frigate: it doesn't run
  expensive recognition on every frame, only on frames where object
  detection already found a person or vehicle.
- Outputs: bounding boxes, class, track ID, timestamp, source camera.

### 6.3 Face ID Module
- Embedding-based (InsightFace/ArcFace or equivalent), not simple
  classification — allows unlimited enrollment without retraining.
- **From Frigate:** label-and-learn UX — unrecognized faces land in an
  "unknown" bucket the operator can label after the fact, which then
  improves future matches. Don't require pre-enrollment before the system
  is useful.
- Output: face embedding, matched identity (or "unknown-#id"), confidence.

### 6.4 Plate & Vehicle Re-ID Module
- Plate path: detect plate region -> OCR (PaddleOCR/EasyOCR).
- Vehicle re-ID path (for motorcycles/obscured plates): embedding on the
  vehicle itself (shape/color/type) so repeat visits are still detected
  without a readable plate.
- Both paths write to the same "vehicle identity" record — a vehicle can be
  matched by plate OR by visual signature, whichever is available on a
  given pass.

### 6.5 Audio Module
- Event classification (music/engine/voice/animal/other) via a pretrained
  audio tagging model (YAMNet/PANNs class).
- Loud-duration tracking: decibel + duration threshold, modeled on the
  NoiseAware/Minut approach — this is simpler and more legally comfortable
  than full classification, and should be the PoC's primary audio feature.
- Voice transcription: **module exists but disabled by default**, gated
  behind an explicit config flag and pending legal review (§9).

### 6.6 Backfill Module
- Batch job: walks a directory of existing photos/video, extracts frames,
  runs the same Detection/Face/Plate modules used for live ingestion, and
  writes to the identity store with source-file metadata instead of a live
  camera ID.
- Must share code with the live path — not a separate parallel pipeline —
  so improvements to detection accuracy apply to both.

### 6.7 Identity & Event Store
- Vector store (FAISS/Qdrant-class) for face and vehicle embeddings.
- Relational/document store for events, timestamps, camera/site metadata,
  and human-applied labels.
- Guest-correlation hook: cross-reference an "unknown vehicle detected"
  event against booking/check-in data (already available from the Kai
  Apartments site's iCal sync) to suppress false alarms during known
  guest arrival windows.

### 6.8 Central Dashboard / Multi-Site Module (V1, stubbed in PoC)
- Allow-list management per site ("Safe List," Flock's term for the same
  concept).
- Cross-site identity matching (same face/vehicle seen at two properties).
- Alerting, retention policy configuration, audit log of who queried what.

## 7. Job Execution Backend (the modularity requirement)

This is the piece that makes "distributed or on-demand" a config choice,
not an architecture decision.

Every processing module (detection, face ID, plate/re-ID, audio) is written
as a plain function/class with a `process(job) -> result` interface —
identical to the Ray actor shape already prototyped. The execution backend
wrapping it is swappable:

| Backend | When to use | Notes |
|---|---|---|
| `LocalBackend` | PoC, single laptop, debugging | Synchronous, no queue — calls the module directly in-process |
| `CeleryBackend` | Multiple intermittently-available machines | Redis-backed durable queue; jobs wait safely if all workers are offline |
| `RayBackend` | Machines that are reliably on and want warm-loaded models | Actor-per-model-type, best throughput when nodes are stable |

A single `JobDispatcher` interface (`submit`, `submit_and_wait`,
`submit_batch`) sits in front of whichever backend is configured, so
application code (the ingestion service, the backfill job) never changes
when you move from "one laptop" to "three intermittent Macs" to "an
always-on fleet." This directly reuses the actor/dispatcher pattern already
built for the Ray PoC — the Celery and Local backends implement the same
interface.

## 8. Non-functional requirements

- **NFR1 — Intermittent connectivity:** edge components must buffer locally
  and sync opportunistically; no component may assume the uplink or a given
  worker machine is always reachable.
- **NFR2 — Data locality/privacy:** raw video/audio never leaves the site
  by default; only metadata, crops, and embeddings are sent upstream.
- **NFR3 — Config-driven, not code-driven:** per-camera zones, masks, and
  which modules run on which camera should be a config file, not a code
  change (Frigate's model).
- **NFR4 — Update rollout:** worker processes must self-update on startup
  (git pull / version check) so intermittently-online machines catch up
  automatically rather than requiring manual push-and-confirm.
- **NFR5 — Legal/compliance:** any module touching biometric or
  conversational-audio data must be off by default and require explicit,
  documented enablement per deployment.

## 9. Risks & open questions

- **Voice transcription legality**: Costa Rica's data-protection law
  (Ley 8968) and general expectation-of-privacy norms make transcribing
  passers-by's conversations meaningfully riskier than logging plates or
  noise levels. Needs local legal review before this module is enabled
  anywhere, including the PoC.
- **ONVIF inconsistency**: real-world Hikvision/Dahua/UniFi implementations
  diverge enough from the ONVIF spec that per-brand adapters are safer than
  assuming one universal integration will work.
- **Total addressable market**: Costa Rica alone is a small market for a
  future B2B product — treat it as a reference deployment, not the target
  market size.
- **Guest-facing disclosure**: any camera/audio monitoring at a rental
  property needs clear signage/listing disclosure, both for compliance and
  guest trust.

## 10. PoC scope (recommend building first)

1. UniFi adapter only (already installed hardware).
2. Detection module: person/vehicle/bicycle/animal via YOLO.
3. Plate OCR + vehicle re-ID for vehicles without readable plates.
4. Face ID with label-and-learn workflow.
5. Audio: loud-duration tracking only (no classification, no transcription).
6. Backfill module against existing photo/video, later NVR, later S3 archive.
7. `LocalBackend` execution only — defer Celery/Ray until the PoC proves
   the detection accuracy is worth distributing.
8. Guest-correlation against existing booking iCal data.

## 11. Future roadmap (V1+)

- Add Hikvision + Dahua adapters.
- Introduce `CeleryBackend` once more than one machine is involved.
- Central dashboard, multi-site identity matching, allow-lists.
- Model-improvement loop (Frigate+ style): aggregate anonymized detection
  corrections across sites to retrain shared models.
- Legal-cleared voice transcription module, opt-in per deployment.

## 12. Success metrics (PoC)

- False-positive rate on "unknown vehicle" alerts during known guest
  check-in windows (target: near zero via booking correlation).
- Face/plate re-identification accuracy on a rolling 2-week backfill test
  set.
- Percentage of vehicle events successfully matched (by plate OR re-ID)
  vs. left fully unidentified — this is the metric that specifically
  validates the motorcycle/obscured-plate use case.
