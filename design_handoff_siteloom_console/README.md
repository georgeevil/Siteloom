# Handoff: Siteloom operator console

## Overview
A dark operations console for Siteloom — the security operator's view over events linked to
people, the media library, enrolled identities, custom detection classes, and the training-data
(library review) workflow. It replaces the current plain Jinja pages
(`siteloom/web/templates/*.html`) with a single-shell, three-pane console.

Target repo: **georgeevil/Siteloom**, branch `main`. The existing UI is FastAPI + Jinja2
(`siteloom/web/app.py`, `siteloom/web/library_routes.py`, `siteloom/web/templates/base.html`).

## About the design files
`Siteloom Console.dc.html` (+ `support.js`) in this bundle is a **design reference created in
HTML** — a prototype showing intended look and behaviour, not production code to copy.
Open it directly in a browser to click through it.

The task is to **recreate these screens in Siteloom's own environment**. Two viable routes:

1. **Stay server-rendered (lowest friction).** Keep FastAPI + Jinja2. Replace
   `templates/base.html` with the console shell (sidebar + top bar), and re-cut
   `index.html`, `library.html`, `identities.html`, `classes.html`, `training.html`
   as the panes described below. Progressive enhancement (htmx or a little vanilla JS)
   covers selection state and the drawers. Recommended if the app stays single-operator.
2. **Introduce a JS front end** (React/Vite served from `siteloom/web/static`) talking to
   JSON endpoints alongside the existing HTML routes. Worth it only if you want live
   updating event lists and heavy multi-select labelling.

Either way: **do not ship the prototype's inline styles as-is.** Lift the token values below
into whatever styling layer you choose.

## Fidelity
**High fidelity.** Colours, type, spacing, radii, and states below are exact and intended to be
matched. Imagery is deliberately placeholder (striped grey blocks) — real thumbnails come from
`/media/{path}` (crops, `LibraryItem.thumb_path`).

---

## Screens / Views

### Shell (all screens)
- Root: `display:flex; height:100vh; overflow:hidden`.
- **Sidebar** 236px fixed, `background #262a2f-equivalent (oklch(0.205 0.009 250))`,
  1px right border `oklch(0.28 0.01 250)`.
  - Logo block: 3×3 grid of 5px squares (alternating `oklch(0.72 0.16 200)` / `oklch(0.4 0.02 250)`),
    2px gap; wordmark "Siteloom" 15px/700; version pill 9px mono.
  - Site selector button: full width, 8px 10px padding, radius 6, 1px border
    `oklch(0.3 0.01 250)`, green pulsing 6px dot (`@keyframes slpulse`, 2s infinite),
    site name 12px/600, camera count 10px mono right-aligned.
  - Nav items: 9px 10px, radius 6, 13px/500 label, a 22px monospace 9px code chip
    (EV/MD/ID/CL/TR) and a right-aligned count. Active = background
    `oklch(0.29 0.05 200)`, chip background `oklch(0.72 0.16 200)` with
    `oklch(0.18 0.03 240)` text. Hover = `oklch(0.26 0.01 250)`.
  - Footer: operator avatar (26px circle, initials), name 12px/600, role 10px muted;
    a device-preview toggle button (prototype-only affordance — see Mobile).
- **Top bar** 56px, bottom border `oklch(0.28 0.01 250)`, background `oklch(0.19 0.008 250)`:
  screen title 15px/600, mono 11px subtitle, right-aligned 230px search field
  (placeholder "Search events, people, plates") and a mono clock chip.

Route mapping: Events → `/`, Media library → `/library`, Identities → `/identities`,
Classes → `/classes`, Training data → `/training`. (`/jobs` and `/noise` are not designed yet —
keep them on the old templates or add nav entries in the same style.)

---

### 1. Events (`/`) — three layout variants
A segmented control (Triage / Timeline / Wall) switches IA. **Ship one**; the other two are
explorations. Recommendation: **Triage** as default, Timeline as a secondary tab.

Filter bar (all variants): 12px 20px padding, background `oklch(0.185 0.008 250)`.
Pill chips: 5px 10px, radius 20, inactive `transparent / border oklch(0.31 0.01 250) / text
oklch(0.74 0.008 250)`, active `background oklch(0.29 0.06 200) / border oklch(0.55 0.1 200) /
text oklch(0.9 0.08 200)`. Chips: Needs review, Unmatched, People, Vehicles, Behaviour.
Right side: mono 11px "N of M events · K filters".

**Variant A — Triage (default).** Table left, detail rail right (372px).
- Row grid, 52px tall, 12px gap, 0 14px padding, bottom border `oklch(0.24 0.009 250)`,
  2px left border coloured by status.
- Responsive tiers driven by *pane* width (window − 236 sidebar − 372 rail):
  - pane ≥ 770px: `78px minmax(180px,1fr) 132px 118px 74px 92px` — Time, Detection, Camera, Identity, Conf, Status.
  - pane ≥ 600px: `78px minmax(150px,1fr) 104px 100px 84px` — Conf column drops, confidence renders inline beside the class name.
  - below: `70px minmax(130px,1fr) 104px 84px` — Camera column also drops (camera prepended to the detection subline) and the detail rail becomes an **absolute overlay drawer** (340px, shadow `-18px 0 44px oklch(0.11 0.01 250 / .55)`) with a ✕ that closes it.
- Row content: mono 11.5px time; 54×34 thumbnail (radius 3); class 12.5px/600 + 11px muted
  subline; camera 12px; identity 12px/500 (unmatched → `oklch(0.82 0.12 30)`);
  confidence 42px bar (4px tall, radius 2, fill `oklch(0.72 0.16 200)`, amber below 0.75)
  + mono 10px value; status chip (mono 9.5px uppercase, 3px 7px, radius 3).
- Detail rail: 16:9 clip placeholder with overlaid mono chips; class 16px/700 + status chip;
  note 12px; a 2-column fact grid (1px gaps over `oklch(0.28 0.01 250)` = hairline table);
  linked-identity card with Open button → identity screen; primary "Clear event",
  outlined destructive "Escalate", dashed "Send crop to training data →".

**Variant B — Timeline.** Per-camera lanes, 58px tall, 150px label gutter, 2px centre rule,
events as absolutely-positioned pills (left % = minutes since 07:00 / 120), clamped 4–96%.
Selected event detail renders in a card below the lanes.

**Variant C — Wall.** `repeat(auto-fill,minmax(268px,1fr))`, 14px gap. Card = 16:10 media block
with status chip + time overlay, then class + confidence, camera, and a footer row with a 24px
identity thumb.

Status tones (background / text / accent):
- New `oklch(0.3 0.06 200)` / `oklch(0.85 0.1 200)` / `oklch(0.72 0.16 200)`
- Reviewing `oklch(0.3 0.05 85)` / `oklch(0.87 0.1 85)` / `oklch(0.75 0.14 85)`
- Flagged `oklch(0.31 0.08 30)` / `oklch(0.86 0.12 30)` / `oklch(0.68 0.16 30)`
- Cleared `oklch(0.27 0.02 250)` / `oklch(0.72 0.01 250)` / `oklch(0.38 0.01 250)`

Data: `Event`, `Detection`, `EventIdentity`, `Camera` from `siteloom/store/models.py`;
the existing `/` route already supplies camera/class/since/until filters and paging.

### 2. Media library (`/library`)
Tab chips (Cameras / Clips / Snapshots / Exports) + retention summary. Grid
`repeat(auto-fill,minmax(232px,1fr))`, 14px gap. Card: 16:9 media block with a state chip
(dot + `live`/`degraded`/`offline`; green `oklch(0.72 0.16 145)`, amber, red) and a resolution
chip; body has name 13px/600, zone 11px muted, and two mono count chips.

### 3. Identities (`/identities`)
Role filter chips (All / Staff / Contractor / Vendor / Unenrolled) + primary "Enrol identity".
Card grid `minmax(300px,1fr)`: 52px square thumb, name 13.5px/600, role chip, org 11.5px muted,
mono footer "N samples · last seen". Role chip tones: Staff cyan, Contractor amber,
Vendor neutral, Unenrolled red (values in Design tokens).
Right rail (340px): 64px thumb + name/org; "Enrolled samples · N" with a 4-column square grid;
recent events list (hairline-separated rows, click → Events); dashed
"Add training samples →" button.
Data: `Identity` (`label`, `vector_count`, `appearance_count`, `best_crop_path`, `last_seen`),
plus `Annotation` rows attributed to the identity (already loaded in `identity_detail`).

### 4. Classes (`/classes`)
Header: "Detection classes", mono model line, primary "New class".
Table (radius 8, 1px border): columns `1fr 116px 128px 96px 64px` = Class / Samples / Threshold /
Precision / Active. Row 56px. Class cell: 10px colour square (radius 2) + name 12.5px/600 +
11px muted description. Threshold = 4px bar + mono value. Active = 34×19 pill toggle
(15px knob, `oklch(0.72 0.16 200)` when on).
Banner below: unverified-examples note + "Open training data".
**New class panel** (352px right drawer): name input, behaviour textarea, trigger-type segmented
(Instant / Sequence / Dwell), camera scope chips, threshold range input
(`accent-color: oklch(0.72 0.16 200)`), an amber advisory box, and
"Create & label samples" → Training data.
Data: `CustomClass` (`name`, `parent_class`, `threshold`, `example_count`) —
per `siteloom/identity/classes.py` these are **k-NN over the shared appearance embeddings**,
so copy must never imply epochs, model training, or a validation run. Verified examples take
effect immediately; `CustomClassifier.rebuild()` is the only "recompute" action.

### 5. Training data (`/training`) — the main new surface
Three panes.

**Left, 262px — library sources.** One card per `LibrarySource`: name 12.5px/600, path 9.5px
mono, 4px progress bar (indexed / total), then two mono rows: "N items / M pending" and
"K failed / last indexed". Selected card border `oklch(0.6 0.1 200)`. "Import" button in the
section header. A dashed footnote explains resumable indexing.

**Centre — crop grid.** Filter chips (Needs review / Verified / Rejected / Unenrolled / Faces /
Vehicles / All) in a horizontally scrolling strip (never wrap); right side has the crop count,
a grouping toggle (By proposed name / Flat) and S/M/L size buttons.
Grid `repeat(auto-fill,minmax(78|112|156px,1fr))`, 9px gap. Tile = 1:1, radius 5, 2px border:
- top-left state chip — proposed (amber) / verified (green) / rejected (red);
- top-right 15px selection checkbox (cyan fill + ✓ when selected);
- bottom-left mono confidence;
- bottom-right 7px green dot when already enrolled;
- filename caption 9px mono below.
Groups get a heading: name 13px/700 + mono "N crops · M verified".

**Right, 326px — labelling inspector.** Header "N selected" with Select all / Clear.
Empty state is a dashed explainer. With a selection: identity list (each row shows
`vector_count`/20 and remaining slots, amber when the gallery is full — see `enroll.py`
`max_vectors`), custom-class chips, a summary box ("N crops → label", "N face vectors will be
added"), then **Verify & enrol** (primary) and **Reject** (outlined destructive).

Responsive tiers for this screen:
- window ≥ 1344px: all three panes.
- 1082–1343px: sources rail collapses into a horizontal source-tab strip above the filter bar (tabs + Import button).
- < 1082px: inspector also becomes an absolute overlay drawer, shown only when crops are selected, with a ✕ that clears the selection.

Data: `LibrarySource`, `LibraryItem` (`status` pending/indexed/failed, `attempts`, `error`,
`thumb_path`), `Annotation` (`bbox`, `class_name`, `confidence`, `source` auto|human,
`verified`, `rejected`, `enrolled`, `crop_path`, `proposed_name`, `identity_id`, `custom_class`).
Verify & enrol maps to `enroll_annotation()`; Reject sets `rejected=True`.

### 6. Import labelled library (modal, from Training data)
880px max width, radius 10, header with 3-step indicator, footer with contextual
back/next labels and a mono status note.
1. **Source** — path input (mono), source-kind segmented (Directory / Google Takeout /
   Immich export), "Identify while indexing" toggle with explanation, two read-only fact
   cards (frames per video, batch size). Footer note: "Nothing is decoded during scan".
2. **Scan** — 4 stat cards (Added / Changed / Skipped / Now pending, 17px/700 numbers) and a
   sample file list (kind chip + path + size). Mirrors `ScanResult`.
3. **Index** — 8px progress bar, "624 of 1,840 indexed / ~11 min remaining", 4 stat cards
   (Processed / Crops found / Failed / Remaining) and an amber note that failed items stay in
   the failed queue for `process(retry_failed=True)`. Mirrors `ProcessResult`.
4. **Done** — green ✓ disc, "Source added", summary line.

### Mobile (390px)
Prototype-only overlay showing the triage list on a phone: sticky header (mono status row,
"Events" 19px/700, flagged count pill), scrolling event cards (64×52 thumb, class 14px/700,
time, camera, identity, status chip, min-height 76px), and a 4-tab bottom bar at 60px
(16px square glyph placeholder + 11px label). Real implementation should hit these targets at
`max-width: 480px`; the desktop three-pane layouts collapse to single column with the detail
rail as a full-screen sheet.

---

## Interactions & behaviour
- Nav switches screens; no page transition animation.
- Selecting an event sets the detail rail (and opens the drawer at narrow widths).
- Event filter chips are additive booleans; the Needs-review chip filters out Cleared.
- Timeline pills and identity "recent events" rows both deep-link into the event detail.
- Training crops toggle selection on click; Select all applies to the *filtered* set.
  Verify & enrol flips crops to verified+enrolled and clears the selection; Reject flips to
  rejected. Both should be optimistic with a server round-trip.
- Class Active toggles are immediate.
- Modals: step-forward/back only, Escape and ✕ close. Nothing auto-advances.
- Panels animate in with `slrise` — `opacity 0 → 1`, `translateY(6px) → 0`, 160–200ms ease.
- Live indicator uses `slpulse` — opacity 1 → .35 → 1 over 2s, infinite.
- Hover states: nav `oklch(0.26 0.01 250)`; rows `oklch(0.235 0.01 250)`;
  cards border → `oklch(0.5 0.06 200)`; primary buttons → `oklch(0.79 0.14 200)`;
  outlined buttons → border `oklch(0.5 0.03 250)`.
- All responsive tiering is computed from measured **pane** width, not viewport width —
  a three-pane layout must subtract its fixed rails before choosing a column set. Reproduce
  that logic (container queries are the clean CSS equivalent).

## State
`screen`, `eventsVariant`, `selectedEventId`, `detailDrawerOpen`, `eventFilters{}`,
`selectedIdentity`, `identityRoleFilter`, `classActive[]`, `newClassPanelOpen`,
`trainingSource`, `trainingFilter`, `thumbSize`, `grouping`, `cropSelection{}`,
`assignIdentity`, `assignCustomClass`, `importOpen`, `importStep`, `importKind`,
`identifyWhileIndexing`, plus a viewport-width subscription.

## Design tokens
Colours (oklch; convert as needed):
- Canvas `oklch(0.17 0.008 250)` · Panel `oklch(0.205 0.009 250)` · Raised `oklch(0.215 0.009 250)`
- Header `oklch(0.19 0.008 250)` · Toolbar `oklch(0.185 0.008 250)`
- Hairline `oklch(0.28 0.01 250)` · Row rule `oklch(0.24 0.009 250)` · Control border `oklch(0.3 0.01 250)`
- Text `oklch(0.95 0.005 250)` · Muted `oklch(0.66 0.008 250)` · Faint `oklch(0.58 0.008 250)`
- Accent cyan `oklch(0.72 0.16 200)` (hover `oklch(0.79 0.14 200)`, on-accent text `oklch(0.16 0.03 240)`)
- Amber `oklch(0.75 0.14 85)` · Red `oklch(0.68 0.16 30)` · Green `oklch(0.72 0.16 145)`
- Class swatches: person cyan 200, vehicle 260, package 145, loitering 85, tailgating 30, PPE 320 (all `0.72 0.16 h`)

Type: **Archivo** (400/500/600/700) for UI, **JetBrains Mono** (400/500/700) for data, times,
paths, IDs, counts and uppercase micro-labels (9–9.5px, letter-spacing .06–.07em).
Scale: 9 / 9.5 / 10 / 11 / 11.5 / 12 / 12.5 / 13 / 13.5 / 14 / 15 / 16 / 17 / 19px.
Radii: 2 (swatch) / 3 (chip) / 5 (thumb) / 6 (control) / 7–8 (card) / 10 (modal) / 20 (pill) / 50% (avatar).
Spacing: 2 / 4 / 6 / 7 / 9 / 11 / 12 / 14 / 16 / 18 / 20 / 22px.
Shadows: modal `0 24px 70px oklch(0.1 0.01 250 / .6)`; drawer `-18px 0 44px oklch(0.11 0.01 250 / .55)`.
Scrollbar: 10px, thumb `oklch(0.33 0.01 250)` with a 3px canvas-coloured border.

## Assets
None shipped. Every image is a striped placeholder
(`repeating-linear-gradient(135deg, rgba(255,255,255,.05|.06) 0 4–7px, transparent → 8–14px)`
over `oklch(0.24–0.28 0.01 250)`). Replace with real crops/thumbnails from `/media/{path}`.
The logo mark is a 3×3 grid of 5px squares — no SVG needed. No icon set is used; the nav uses
2-letter monospace codes deliberately. If you adopt an icon library, keep the type scale.

## Files
- `Siteloom Console.dc.html` — the full prototype (all five screens, both modals, mobile overlay).
- `support.js` — runtime required to open the prototype locally. Not part of the design.
- Upstream references: `siteloom/web/app.py`, `siteloom/web/library_routes.py`,
  `siteloom/web/templates/base.html`, `siteloom/library/indexer.py`,
  `siteloom/identity/enroll.py`, `siteloom/identity/classes.py`, `siteloom/store/models.py`.

## Known gaps
- Events and Media content is invented; only Training data, Classes and Identities are
  grounded in repo models. Confirm real field names before wiring.
- `/jobs` and `/noise` have no design yet.
- Empty, loading, and error states are not designed (except the failed-items note in import).
- No auth/permission states.
