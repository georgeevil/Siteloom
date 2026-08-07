# Identity management, matching confidence & event noise — analysis

*2026-08-07 · consultation pass over the identity layer, the console UI, and the ingest
pipeline, prompted by three linked observations on the live Kai deployment: false-positive
identity matches with no way to correct them, noisy bursts of short low-confidence person
events, and no operator control over similarity thresholds or matching models.*

*Filed as Linear milestone **M1.5 — Identity accuracy & noise reduction** (CLD-36 … CLD-46).
UI claims below were verified in a live browser session against a seeded scratch database.*

## 1. The three problems, traced to code

### 1.1 Relationships can be judged but not managed

The console can *rename* an identity, *merge* two identities (properly — vectors are
re-pointed, `library_routes.py:598`), and record *verdicts* on event↔identity claims
(CLD-16). It cannot:

| Operation | Status |
|---|---|
| Attach an identity to an event ("it was actually Alice") | missing — no endpoint, no picker (`_event_rail.html`, `event.html`) |
| Unlink / void a wrong `EventIdentity` match | missing — `verdict="wrong"` leaves the name rendering forever |
| Reassign a link to another identity | missing — though merge already contains the SQL |
| Split an identity for real | stub — `library_routes.py:657-689` re-points annotations only; **no vectors move**, despite its docstring |
| Delete an identity / remove single vectors | missing |
| Cross-link a face identity with a person identity | deferred to V1 by design (`models.py:138`) |

The asymmetry is the core UX problem: the system creates relationships automatically and
the operator can only *comment* on them. → **CLD-36** (link/unlink/reassign), **CLD-37**
(real split).

### 1.2 Noise and false positives share one root: the resolver and the stitcher are naive

`IdentityResolver.resolve()` (`identity/resolver.py:51-132`) is a single
nearest-neighbor lookup against a fixed per-identifier threshold:

- **No k-NN vote, no first-vs-second margin.** One borderline vector in the wrong gallery
  wins outright. (Ironically `identity/classes.py:51-87` already implements a
  mean-of-hits vote — for custom classes only.)
- **No temporal or camera context.** "This camera matched identity 42 four seconds ago"
  carries zero weight.
- **Every sub-threshold crop unconditionally creates a new Identity *and* enrolls its
  embedding** (`resolver.py:75-97`). A single blurry crop becomes a permanent magnet in
  vector space that attracts the next blurry crop. This is exactly the "many similar
  unknown persons minutes apart" being observed.

On the event side (`ingest.py`):

- `_stitch_event` tries **one** candidate — the newest event in the gap window
  (`ingest.py:505`). Two people in frame → the newest event is the other person → IoU
  fails → new fragment.
- `person` is in no class group (`class_groups` default covers vehicles only).
- Nothing merges two adjacent events linked to the *same identity*.
- Only max-confidence is aggregated (`Event.best_confidence`) — one lucky 0.55 frame is
  indistinguishable from sustained 0.55.
- `EVENT_LINK_GAP_S = 120` is hard-coded.

→ **CLD-40** (event fragmentation), **CLD-41** (resolver: vote + margin + recency +
provisional identities). These two are the highest-leverage fixes: they reduce the volume
*and* raise per-event confidence, which is precisely the "fewer, better events to
supervise" goal.

### 1.3 Tuning exists in config but not in the operator's hands

- Per-identifier thresholds (`identity.identifiers.<key>.threshold`; face 0.36,
  person 0.80, vehicle 0.82) are consulted on every resolve — but the `/classes` page
  renders them as **read-only bars**, even though `POST /classes/detection` *already
  accepts* threshold writes and persists to YAML. The UI just never sends the field
  (`classes.html:320`). → **CLD-38** — wire the sliders and add a dry-run preview
  computed from stored `EventIdentity.similarity` ("at 0.42, N recent matches flip").
- Per-camera overrides (`EventRulesOverride`, `config.py:32`) stop at four event fields;
  the identify gates (`identify_min_confidence` / `identify_min_crop_px` /
  `identify_only_significant`), `stitch_min_iou`, and identifier thresholds are site-wide
  only. → **CLD-39**.
- Caveat to surface honestly in the UI: `siteloom serve` and `siteloom run` are separate
  processes; threshold changes reach live ingest on restart.

### 1.4 Models are effectively frozen

- `build_embedder` (`embedders.py:177`) is a closed two-branch factory
  (`face` = YuNet→SFace 128-d, `generic` = ResNet-18 512-d). No third option without code.
- **Nothing rebuilds the face/person/vehicle collections** after an embedder or
  `crop_margin` change — only `class-examples` has a rebuild. A dimension change would be
  rejected raw by Qdrant; a same-dimension change silently mixes incompatible embedding
  spaces. → **CLD-43** (`siteloom identity rebuild` + dimension guard).
- **Bug found during this analysis: `siteloom/training/` is not in the repo** yet is
  imported by `train status|face|export-detector|detector` and `tests/test_training.py` —
  all four commands ImportError today, and with them the documented fine-tune →
  re-threshold loop. → **CLD-42**.
- The requested "try a model on a small subset first" workflow becomes: open the factory
  to a registry, evaluate candidates in *shadow collections* against ground truth the
  system already has (verified annotations + event verdicts), report accuracy *and* speed,
  and only then apply = config change + rebuild + threshold retune. → **CLD-44**, with the
  supervised auto-tuning loop (**CLD-46**) deliberately last and gated on a success
  criterion.

## 2. How comparable tools handle this (and what to borrow)

| Tool | Mechanism | Siteloom takeaway |
|---|---|---|
| **Immich** | One "max recognition distance" slider with recommended bounds; guidance: *err strict and merge later, split is unsalvageable*; changing model/settings requires re-running the facial recognition job (the UI says so); merge/unassign face in UI; re-cluster "All" vs "Missing" | Threshold slider + preview (CLD-38); honest restart caveat; unlink/reassign (CLD-36); rebuild as an explicit job (CLD-43); adopt the merge-over-split guidance in Help (CLD-45) |
| **Double Take** | Per-detector `match` / `unknown` confidence + min-area; train/untrain from the UI | Per-camera thresholds (CLD-39); the training review page already covers train/untrain |
| **Frigate** (face recognition) | A name is only assigned when recognized *consistently across frames* (area-weighted running score); training guidance: label the clear low-scoring crops, not the 90%+ ones | Resolver consistency: vote + margin + provisional identities (CLD-41); surface *borderline* crops for supervised labeling — the highest-value queue |
| **FiftyOne / autodistill** | Evaluate candidate models on a curated labeled subset before adopting; big model supervises small model | Shadow evaluation (CLD-44); supervisor-model audit loop as proposals, never mutations (CLD-46) |

Sources: [Immich facial recognition docs](https://docs.immich.app/features/facial-recognition/),
[Immich better-clusters guide](https://docs.immich.app/guides/better-facial-clusters/),
[Frigate face recognition docs](https://docs.frigate.video/configuration/face_recognition/),
[Double Take](https://github.com/jakowenko/double-take).

## 3. Recommended order of work

1. **CLD-41 resolver consistency** and **CLD-40 event de-fragmentation** — cut the noise
   at the source; everything downstream (review, labeling, tuning) gets cheaper.
2. **CLD-36 link/unlink/reassign** and **CLD-37 real split** — make every remaining
   mistake correctable, so supervision compounds instead of leaking.
3. **CLD-38 threshold sliders + dry-run** and **CLD-39 per-camera overrides** — the
   requested back-and-forth experimentation, safely.
4. **CLD-42 training package** (bug), **CLD-43 rebuild**, **CLD-44 model trial harness** —
   unlock model experimentation.
5. **CLD-46 supervisor auto-tuning** — only if its disagreement flags demonstrably enrich
   for wrong matches on a held-out verdict set.
6. **CLD-45 Help tab** — can land any time; update its flowcharts as the above ship.

## 4. UI review notes (browser session)

Captured against a seeded scratch DB (2 cameras, 11 events reproducing the burst pattern,
9 unknown-person identities, mixed verdicts). Screenshots attached to the relevant Linear
issues. Confirmed in the rendered UI:

- Events list: the burst reads as 7 separate person rows within ~15 minutes, several
  under 0.66 confidence; two more only appear with the ephemeral chip on. The identity
  column shows a different `unknown-person-N` per row — the pileup is visible exactly as
  reported.
- Event rail: identity claims offer ✓ / ✗ / miss only; no picker, no unlink.
- `/identities`: unknown cards dominate the grid; merge/split is only reachable through
  the detail page's collapsed section.
- `/classes`: identifier thresholds render as static bars; event rules are editable,
  thresholds are not.
