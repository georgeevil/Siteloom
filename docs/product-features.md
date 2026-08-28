# Siteloom — feature inventory for product review

*Snapshot of `main` on 2026-08-28. One line per feature, grouped by what an operator would call it; each line says whether it is **shipped**, **flagged** (built, off by default), or a **gap** (filed, not built) with the Linear id. Surface names are the console routes and CLI commands as they exist today. The PRD (`video-processing-platform-prd.md`) is the requirements document; this is the "what does it actually do" companion.*

**Legend:** ✅ shipped · 🚩 flagged off by default · ⚠️ shipped with a known limitation · ⛔ gap (filed)

---

## 1. Capture — getting video in

| Feature | Status | Notes |
|---|---|---|
| UniFi Protect cameras (live RTSPS) | ✅ | `adapter: unifi`; NVR bootstrap resolves stream URLs, self-signed cert tolerated (`verify_ssl`) |
| Generic RTSP cameras | ✅ | `adapter: rtsp`, anything OpenCV/FFmpeg decodes |
| Local files / archives as a "camera" | ✅ | `adapter: file` — dev path and the seed of backfill |
| Frigate as a detector front-end | ✅ | `siteloom frigate`: MQTT `frigate/events` → snapshot from Frigate's API → identity → republish |
| Concurrent cameras with reconnect | ✅ | one worker thread per camera, reconnect on drop (CLD-14) |
| Per-camera sample rate, zones, masks | ✅ | YAML, never code (`sample_fps`, `zones`, `masks`) |
| Historical NVR backfill (UniFi) | ✅ | `siteloom backfill-unifi` and the `/backfill` screen: NVR motion/smart-detect windows by default, `--chunk-minutes` full sweep; resumable; dedupes by NVR event id |
| Archive backfill (directory of clips) | ⚠️ | `siteloom backfill` reuses the live pipeline but is not yet a tracked/resumable job (CLD-12) |
| Drop-and-reindex a recent window | ✅ | `POST /jobs/reindex` — purge + re-run through current settings |
| Live view wall | ✅ | `/live`: shared-reader MJPEG, one RTSP reader per camera regardless of viewers, idle-stopped; ends cleanly on shutdown (CLD-132) |
| Run live ingest and the console in one process | ⛔ | embedded Qdrant is one client per path, so `run` + `serve` cannot coexist; near-live today is periodic `/backfill/start` sweeps inside `serve`. Qdrant server URL is CLD-300 |

## 2. Detection & tracking

| Feature | Status | Notes |
|---|---|---|
| YOLO object detection | ✅ | ultralytics; classes and per-class confidence from config (`detection.classes`, `/classes`) |
| Apple Silicon (`mps`) + CoreML/ONNX runtimes | ✅ | primary target; device is config |
| Multi-object tracking | ✅ | **BoT-SORT + ReID on detector features** (shipped 2026-08-25, corpus-measured); `track_buffer` derived from `track_buffer_s × sample_fps` |
| Zone hit-testing | ✅ | bbox bottom-centre point (Frigate convention) |
| Event de-fragmentation (3 layers) | ✅ | track fast-path, IoU stitching over recent events, identity-aware merge (CLD-40); all in frame time |
| Significance gate | ✅ | short/low-confidence events are stored but not identified until they earn it; warm-up frames replayed on the flip (CLD-286) |
| Occlusion handling | ✅ / 🚩 | per-camera `OcclusionMonitor`: `Detection.occluded` (match but never learn), `suspect_birth`, `Event.suspect_swap` freeze + operator verdict. `events.occlusion_stitch` is **off by default** |
| Per-camera detection settings | ✅ | `CameraConfig.detection` override merges over site (CLD-101); `crop_margin`/`classes`/`device` deliberately not overridable |
| Automatic day/night profiles | ✅ | measured per frame from saturation (`scene.py`), `CameraConfig.night` layers over day (CLD-129); separate model+tracker per profile |
| Tracker A/B harness | ✅ | `scripts/track_ab.py check` over `docs/testing/tracker-corpus.md` clips — tracker changes are measured, not argued (CLD-98) |
| Custom sub-classes (k-NN on crops) | ✅ | `siteloom classes add/list/rebuild`, `/classes`; no training run, examples improve the class immediately |
| YOLO face-detector fine-tuning | ✅ | `siteloom train detector` / `export-detector` — improves *detection* only, not who someone is |
| Partial `detection.tracker:` override re-enables `fuse_score` | ⛔ | site-level dict replacement trap (CLD-258) |
| Corpus coverage for IR and mixed-class scenes | ⛔ | CLD-306 |

## 3. Identity — who and what

| Feature | Status | Notes |
|---|---|---|
| Face identification | ✅ | YuNet detect → align → SFace 128-d (OpenCV built-ins, weights pinned by digest, CLD-50); own cosine threshold (~0.36) |
| Person / vehicle / any-class re-identification | ✅ | shared ResNet-18 appearance embedding (512-d); auto-added identifier for any new class (`identity.auto_add_classes`) |
| Vector store | ✅ | embedded Qdrant, one collection per identifier; same client speaks to a remote server later |
| Label-and-learn (no pre-enrollment) | ✅ | identities start as `unknown-…`; labelling renames past and future matches |
| Consistency gating | ✅ | threshold **and** margin over the runner-up (`min_margin`), same-camera recency tie-break, pending pool until `min_sightings` (CLD-41) — a resolution may return *no* identity |
| Per-event budgets | ✅ | `learn_max_per_event`, `mint_max_per_event` — one visit cannot mint fifty identities |
| Gated gallery accretion | ✅ | quality floor, per-event learn cap, no learning into a claim ruled wrong, runner-up guaranteed visible to the margin check (CLD-139) |
| Score aggregation per identifier | ✅ | `max` or `mean_top_k` (CLD-152); vehicles run `mean_top_k` live |
| Plate-first vehicle matching | ✅ | plate OCR and visual re-ID write the *same* Identity; a plate beats similarity; a visual match can learn its plate |
| Per-camera identity thresholds | ✅ | `CameraIdentityOverride` (CLD-39), one resolution shared by ingest and replay |
| Identifier defaults merged, never replaced | ✅ | naming `face:` to change one number no longer disables gating (CLD-125); `doctor` prints values in force |
| Embedding-space stamp + rebuild | ✅ | `siteloom identity rebuild` re-embeds from stored crops with labels surviving; `doctor` warns on drift (CLD-106) |
| Vehicle colour fingerprint | 🚩 | `identity.fingerprint.enabled` — pure pixel maths, IR-honest (`no-chroma`, never "gray"), counts not text (CLD-254) |
| Offline replay lab | ✅ | `siteloom lab replay --events N` — re-run real events under candidate settings in a sandbox seeded from live galleries, scored against verdicts |
| `min_sightings` quorum counts frames of one visit | ⛔ | duplicate mint from a single event (CLD-297) |
| Tie-broken and plate matches still teach the visual gallery | ⛔ | CLD-257 |
| Cross-event learning from operator verdicts (vector negatives) | ⛔ | CLD-33 |
| One "Person" over face + body identities | ⛔ | CLD-141 |

## 4. Plates (LPR)

| Feature | Status | Notes |
|---|---|---|
| Plate OCR | ✅ | optional (`requirements-plates.txt`); without it vehicles degrade to visual re-ID with a logged warning |
| Every read stored with its measurements | ✅ | `PlateRead` rows carry text, confidence, box width, sharpness, char confidence, and the floor they were judged against — re-tune by query, never re-run (CLD-85/119) |
| Quality floors, per camera | ✅ | `plate_min_width_px`, `plate_min_sharpness`, `plate_min_char_confidence`; resolved by `plate_floors_for` (CLD-128) |
| OCR rationing per vehicle | ✅ | `plate_ocr_interval_s` in **frame time** (CLD-130); embedding still runs on rationed frames |
| `/plates` per-visit view | ✅ | accepted reads grouped by event with best-known text; per-frame rows kept underneath (CLD-131) |
| Plate search, per-plate history | ✅ | `/plates/p/{plate}`, global search |
| Watchlist alarms | ✅ | `/plates/watchlist`; once per event, to MQTT/webhooks |
| Corrections & verdicts | ✅ | correct a read, confirm/wrong, apply to identity, bulk actions |
| Operator owns the plate | ✅ | clear / edit / move a plate between identities, re-learn lock (CLD-134) |
| Cross-frame consensus gate | ⛔ | accept only when reads agree (CLD-114) |
| Five-way plate verdicts (illegible / not-a-plate / wrong vehicle …) | ⛔ | CLD-138 |
| Plate reads on the Frigate path | ⛔ | consumer records none → watchlist silent there (CLD-117) |

## 5. Audio

| Feature | Status | Notes |
|---|---|---|
| Loud-episode detection | ✅ | dBFS RMS threshold + min duration + release gap → `NoiseEvent`; `/noise` |
| Sources | ⚠️ | file and `backfill-unifi` clips only — a live RTSP stream carries no audio into the pipeline (OpenCV drops it) |
| Classification / transcription | 🚫 | intentionally absent (NFR5, PRD §9) |

## 6. Bookings & the §12 metric

| Feature | Status | Notes |
|---|---|---|
| iCal booking sync | ✅ | `siteloom sync-bookings`; one or many feeds (one per unit), all-day dates placed at site check-in/check-out hours |
| Manual bookings | ✅ | `/bookings` add/edit/delete; feed sync never overwrites a manual row (CLD-90) |
| Guest-window stamp on events | ✅ | `Event.guest_window` from `[check-in − pre, check-in + post]` |
| Alerts that use the stamp | ⛔ | **no alert path exists end to end** — no persisted alert, no suppression consumer, channels disabled live (CLD-255, Urgent) |
| Guest windows reload while `run` is up | ⛔ | loaded at startup only (CLD-68); the sweep path reloads per run |

## 7. Operator console (`siteloom serve`)

| Screen | Status | What it does |
|---|---|---|
| `/` Events triage | ✅ | table + detail rail; timeframe presets, facet chips, column picker, keyset continuous scroll; clear / escalate / verdicts / "more than one person here" |
| `/events/{id}` | ✅ | claims with honest provenance badges (plate / visual / human, similarity vs bar — CLD-136), link / unlink / reassign / bulk, miss recording filed under the identifier that actually ran (CLD-135), swap verdicts |
| `/identities`, `/identities/{id}` | ✅ | label, merge, split (moves vectors with annotations), cover photo re-derived and choosable (CLD-137), plate management, sightings |
| `/plates`, `/plates/p/{plate}` | ✅ | see §4 |
| `/incidents` | ✅ | Escalate opens an Incident that outlives the session: notes, status, export (CLD-96) |
| `/bookings` | ✅ | see §6 |
| `/noise` | ✅ | audio episodes |
| `/live` | ✅ | monitor wall |
| `/search` | ✅ | global search across events, people, plates, library |
| `/stats` | ✅ | per-identifier precision with denominators, similarity histograms (never pooled, plates excluded), threshold trade-offs from raw similarities, "cleared without a verdict" |
| `/training` | ✅ | three-pane annotation review + **Today's queue** of borderline judgments (CLD-8) |
| `/library`, `/library/import` | ✅ | media library with two-phase indexing, import wizard (directory or Google Takeout), backlog banner (CLD-126) |
| `/classes` | ✅ | detection classes & confidence, custom classes, thresholds, site timezone |
| `/detector`, `/detector/tune`, `/detector/help` | ✅ | guided per-camera tuning lab: sandboxed trials on NVR windows/clips/uploads, evidence frames, presets + merge↔split axis, live overlay preview, apply/copy/revert with `config-history/` snapshots, propose-only settings search (CLD-101/102/106) |
| `/train` | ✅ | face fine-tune runs, adopt only on a valid held-out improvement, enroll backlog, vector rebuild |
| `/backfill` | ✅ | NVR backfill console with coverage view (CLD-93) |
| `/jobs` | ✅ | every long-running operation with heartbeat, ETA, cancel, reap; SSE stream |
| `/audit` | ✅ | who did what (username denormalised) |
| `/users` | ✅ | role / disable / revoke sessions (creation is CLI-only by design) |
| Timezone | ✅ | store naive UTC, render site-local for every viewer (CLD-100) |
| Phone layout | ⚠️ | triage cards on a phone (CLD-25 partial) |
| Timeframe picker on every screen | ⛔ | events has it; others do not (CLD-115) |
| Association workbench | ⛔ | dedicated correction screen (CLD-116); bulk reassign / event split (CLD-140) |
| Escalation delivery | ⛔ | Escalate → **email with a link** decided 2026-08-28 (CLD-316); no SMTP config exists yet |

## 8. Training data & library

| Feature | Status | Notes |
|---|---|---|
| Library scan + bounded index | ✅ | `library scan` cheap registration, `library index --limit N` expensive pass; `failed` reported separately from `pending` |
| Google Takeout import | ✅ | two-pass face assignment from per-photo people tags; proposals kept separate from identities |
| Annotations (one table, four jobs) | ✅ | provenance (`auto`/`human`/`import`), verified-by (CLD-95), rejected kept as negatives, boxes normalised |
| Only verified, non-rejected annotations train | ✅ | `training/dataset.py` |
| Face model evaluation guard | ✅ | never adopt without a valid held-out improvement |
| Enrollment sweep | ✅ | `siteloom train enroll` — a label without vectors is a name the system cannot see |
| Send an event crop to training data | ⛔ | undecided half of CLD-28 |

## 9. Integrations & API

| Feature | Status | Notes |
|---|---|---|
| MQTT publish | 🚩 | identity/event payloads, once per event+identity; degrades to a log line (NFR1). Disabled on the live site |
| Webhooks | 🚩 | fire-and-forget; disabled live |
| CompreFace-compatible recognition API | ✅ | `/api/v1/recognition/*` with `x-api-key`; Double Take can point at it; unknown-bucket identities never appear as subjects; degrades to 503 while the store is held (CLD-110) |
| Frigate consumer | ✅ | see §1 |
| Email | ⛔ | CLD-316 |

## 10. Operations, security, manageability

| Feature | Status | Notes |
|---|---|---|
| Service units | ✅ | `siteloom service install/start/stop/status/logs` for launchd and systemd, argv reflected off the CLI, `SuccessExitStatus=130`, no `--workers` |
| `siteloom doctor` | ✅ | every environmental check as a `Check`, never raising; embedding-space drift, identity gating in force, services |
| `/healthz`, `/readyz` | ✅ | read-only, terse, cached (CLD-54) |
| Long-running jobs | ✅ | `ProgressReporter`: heartbeat row, Rich bar or log lines, Ctrl-C/SIGTERM/SIGHUP → finish batch, commit, print resume command; `jobs list/watch/cancel/reap`; stale-run detection with pid-reuse guard (CLD-57) |
| Resumability | ✅ | guarded by the differential harness `tests/test_resume_equivalence.py` |
| Factory reset | ✅ | `siteloom reset` — rows, media and vectors clear together or not at all (CLD-124) |
| Auth | ✅ | turns on with the first `User` row; roles restricted < view < edit < admin; scrypt, revocable sessions, 14-day TTL, login backoff, open-redirect guard |
| Restricted role sees recognition, not identity | ✅ | one server-side substitution applied at render, enforced by a walk over every GET (CLD-111) |
| CSRF by provenance | ✅ | one middleware, `Origin`/`Sec-Fetch-Site`, `/api/v1/` exempt (CLD-58) |
| Audit log | ✅ | every mutating request, username denormalised |
| Media path containment, MQTT input validation, weight integrity | ✅ | CLD-49/48/50 |
| `service stop` durable under launchd KeepAlive | ⛔ | CLD-298 |
| Open mode over the biometric store on the live site | ⛔ | zero users today — first-user creation as part of deployment (CLD-259) |
| Restricted/Viewer cannot sign out | ⛔ | CLD-260 |
| TLS in front of the console / trusted proxies | ⛔ | CLD-112 (proxy-terminated by design) |
| Secrets in plaintext YAML, NVR TLS verify off | ⚠️ | accepted PoC decision (CLD-53) |
| Analysis yielding to live ingest | ⛔ | CLD-105 |

## 11. Multi-site / V1 (PRD §6.8, §11)

| Feature | Status |
|---|---|
| Central dashboard, multi-site | ⛔ not started (stubbed in PRD) |
| Remote Qdrant / shared vector store | ⛔ CLD-300 |
| Celery / Ray job backend | ⛔ `JobDispatcher` API is in place; only `LocalBackend` exists |
| Edge compute for embeddings | ⛔ modules are already serialisable-payload only (NFR2) |

---

### Where the product stands (2026-08-28)

- **Capture → detect → identify → store** works end to end on the Kai site and survives restarts.
- **Accuracy is the open question.** The last operator-judged sample (pre-reset) put visual precision at ~14% on a biased, suspicious-first sample (CLD-256). The gates that were missing then are now in force; the verdict corpus to re-measure with is being regrown (CLD-18 soak).
- **The PoC success metric (§12) has no end-to-end path yet** — CLD-255 is the single most important gap.
- **Everything is single-site, single-process.** That is fine for the PoC and is the whole of M2+.

*Related reading:* `docs/operations.md` (running it), `docs/tuning-workflows.md` (the lab), `docs/identity-management-analysis.md` (why the identity layer looks the way it does), `docs/testing/` (how each guarantee is tested).
