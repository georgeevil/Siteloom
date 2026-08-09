# Lint and types (CLD-82)

```
.venv/bin/ruff check .        # lint      — passes clean today
.venv/bin/ruff check --fix .  # …and fix what is auto-fixable
.venv/bin/mypy                # types     — passes clean today
```

Both are wired into `.github/workflows/ci.yml`. Both pass on the tree as it stands, which
was the design constraint: a gate whose adoption requires touching forty source files is
a gate that lands in conflict with every open branch and gets reverted.

## Ruff

`select = ["E4", "E7", "E9", "F", "C90"]`. Things that are wrong, not things that are
unfashionable — import/syntax/name errors that bite at runtime, plus a complexity cap.

Two settings exist to keep the linter from fighting the house style:

* **`line-length = 100`, and E501 is not selected.** The style runs to long prose comments
  that explain why a rule exists; those paragraphs are load-bearing documentation and
  reflowing them to 88 columns would shred them. Only 5 lines in the tree exceed 100
  (max is 130, `siteloom/store/db.py:155`). The value is for the formatter.
* **`mccabe.max-complexity = 20`, not the default 10.** Seventeen functions sit above 10.
  The tree's real shape is "≤19 everywhere, plus one outlier at 92" — a threshold that
  flags a sixth of the codebase gets switched off within a week; one that flags nothing
  today catches the next function that grows.

`flake8-bugbear.extend-immutable-calls` is configured even though `B` is not selected, so
whoever turns it on does not spend an hour rediscovering that all 15 `B008` hits are the
`typer.Option` / `fastapi.Depends` declarative-default idiom.

### The baseline

`per-file-ignores` in `pyproject.toml` is a **baseline, not policy**. Eleven entries, each
a real finding, each pinned to the single file it lives in so the rule stays live in the
other fifty-odd. Deleting an entry after fixing the code is the ratchet.

| File | Rule | Finding |
| --- | --- | --- |
| `siteloom/cli_library.py:6` | F401 | `logging` imported, unused |
| `siteloom/reindex.py:18,26` | F401 | `json`, `SiteConfig` imported, unused |
| `siteloom/web/library_routes.py:16` | F401 | `fastapi.Query` imported, unused |
| `siteloom/web/app.py:241` | C901 | `create_app` complexity **92** |
| `tests/test_identity.py:10` | F401 | `IdentifierConfig` |
| `tests/test_integrations.py:9` | F401 | `pytest` |
| `tests/test_library.py:12` | F401 | `CameraConfig` |
| `tests/test_noise_page.py:14` | F401 | `pytest` |
| `tests/test_takeout.py:13` | F401 | `pytest` |
| `tests/test_resume_equivalence.py:277` | F821 | `Path` used in an annotation, never imported |
| `tests/test_search_sse.py:141` | E741 | `l` as a loop variable |

Two deserve more than a line:

* **F821 in `test_resume_equivalence.py`** is latent rather than live. The file has
  `from __future__ import annotations`, so `source: Path` is never evaluated and the suite
  passes. It breaks the moment anything calls `typing.get_type_hints` on that function.
  One-line fix: `from pathlib import Path`.
* **C901 on `create_app`** at 92 is the auth/audit middleware plus the whole route table
  in one function. CLAUDE.md is explicit that enforcement belongs in *one* middleware, so
  the fix is extracting the route registrations, not splitting the middleware.

None of the above was fixed here on purpose — this branch owns tooling, and every one of
those files was being edited concurrently on other branches.

### What tightening would flag

Full tree, measured with the config above plus `--extend-select`. Counts are net of the
baseline.

| Family | Hits | Verdict |
| --- | --- | --- |
| `RET`, `LOG`, `G` | **0** | Free. Turn on now. |
| `PTH` | 1 | `PTH123` (`open()` over `Path.open()`). Free. |
| `ISC`, `EXE` | 1, 2 | Trivial. |
| `A` | 3 | 3× `A002` builtin-shadowing argument names. |
| `FURB` | 3 | 3× `re.S` → `re.DOTALL`, auto-fixable. |
| `PERF` | 8 | 5 manual list comprehensions, 3 `.items()` misuse. |
| `B` | 10 | **Highest value per line.** 5× `B904` (`raise … from`), 3× `B007`, and 2× `B023` in `siteloom/web/live.py:212` — a closure capturing the loop variables `feed`/`seen`, which is a real latent bug, not a style note. |
| `C4` | 11 | 8× `C408` (`dict()` → `{}`). |
| `I` | 12 | `I001` unsorted imports, **all auto-fixable**. The natural first tightening: one `ruff check --fix`, zero judgement calls. |
| `SIM` | 14 | 9× `SIM117` nested `with`. Some of those nests are readable as written. |
| `RUF` | 35 | Mixed. Note `RUF003` (7) flags ambiguous unicode *in comments* — that is the em-dash prose style, and enabling it would be exactly the config-fights-the-codebase failure. `RUF100` (4) only makes sense once `B`/`C90` are fully selected. |
| `TRY` | 53 | 50 of them are `TRY003` (long messages in `raise`), which is a style opinion this codebase reasonably rejects. |
| `UP` | 86 | All cosmetic, ~all auto-fixable: 55× `UP017` (`timezone.utc` → `UTC`), 18× `UP037`, 12× `UP024`. Cheap but a large diff — land it alone, on a quiet day. |
| `ARG` | 96 | Unused arguments. Mostly FastAPI/Typer signatures where the argument is the framework's, not ours. Low value. |
| `DTZ` | 106 | **The one with real correctness weight.** 102× `DTZ001`: naive `datetime(...)` construction. For a platform that stamps event times across sites and compares frame time to wall clock, tz-naive datetimes are a latent class of bug. Also the most invasive change on this list — it belongs in its own issue with a decision about whether the store is naive-UTC by contract. |
| `N` | 162 | 126× `N806` non-lowercase locals — largely `Session`, the SQLAlchemy sessionmaker, which is conventionally capitalised. Do not enable. |
| `PL` | 803 | 436× `PLR2004` magic values (mostly in tests) and 297× `PLC0415` import-outside-top-level — the latter is the *deliberate* lazy import of torch/ultralytics that keeps CLI startup fast. Do not enable. |
| `D` | 1258 | Docstring formatting. No. |
| `S` | 1632 | 1615 are `S101` (`assert` in tests). The residue is worth one manual read — `S608` hardcoded-SQL (4) and `S310` url-open (5) — but not as a gate. |
| `ANN` | 2124 | See below. |

Suggested order: `RET`/`LOG`/`G`/`PTH`/`ISC`/`EXE`/`A`/`FURB` (free) → `I` (one autofix) →
`C4`/`PERF` → `B` (fix `B023` first, it is a bug) → `UP` (its own PR) → `DTZ` (its own issue).

## Mypy: not globally, and here is why

The codebase is **not annotated enough for a global gate**:

* **127 errors** at mypy's own defaults with `ignore_missing_imports`, across 21 modules.
* **581 errors** under `--strict`, across **40 of 58** modules.
* Concentrated: `guests.py` alone has 46 default-mode errors (78 of the 127 are
  `union-attr`, overwhelmingly `icalendar`'s optional-returning API), `web/app.py` 17,
  `identity/resolver.py` 13, `backfill.py` 11, `stats.py` 10.

Switching that on globally buys one of two things, neither of which is a caught bug:
months of unrelated churn, or a wall of `# type: ignore`.

### The incremental path, which is what is configured

```toml
[tool.mypy]
files = ["siteloom"]
ignore_missing_imports = true
ignore_errors = true          # analyse everything, report nothing…

[[tool.mypy.overrides]]
module = [...]                # …except here
ignore_errors = false
disallow_untyped_defs = true  # + the rest of --strict, per-module
```

Everything is analysed — imports still have to resolve, so a broken signature in an
unchecked module is still visible where a checked module calls it — but errors are only
*reported* for modules on the seed list. The list only grows.

The seed is the eight modules that are already `--strict`-clean **and** are contracts
other layers are written against:

`siteloom.adapters.file`, `siteloom.adapters.rtsp`, `siteloom.dispatch.*`,
`siteloom.identity.plates`, `siteloom.modules.base`, `siteloom.store`,
`siteloom.training.dataset`, `siteloom.web.nav`

`dispatch` is the point of the exercise: CLAUDE.md promises a Celery/Ray backend will slot
in with zero application-code changes, and that promise is a type contract. A Jinja route
handler is not, which is why the web layer is not on the list.

The seed covers 10 of the 18 strict-clean modules; the other 8 are `__init__.py` shims
with nothing to check.
Near misses worth adopting next, cheapest first:

| Module | Strict errors |
| --- | --- |
| `siteloom/config.py` | 1 |
| `siteloom/identity/registry.py` | 1 |
| `siteloom/adapters/base.py` | 1 |
| `siteloom/integrations/webhooks.py` | 1 |
| `siteloom/web/live.py` | 1 |
| `siteloom/reindex.py` | 2 |
| `siteloom/training/detector.py` | 2 |
| `siteloom/store/db.py` | 3 |
| `siteloom/store/models.py` | 3 |
| `siteloom/modules/detection.py` | 3 |

`config.py` and `store/models.py` are the ones to do first — they define the shapes
everything else passes around, so type discipline there propagates.

### Verifying the gate actually bites

A gate that passes because it checks nothing is worse than no gate. Adding two known-dirty
modules to the seed and re-running reports, as it should:

```
$ mypy --config-file <seed + identity.registry + reindex>
siteloom/identity/registry.py:51: error: Function is missing a return type annotation  [no-untyped-def]
siteloom/reindex.py:55: error: Function is missing a type annotation for one or more parameters  [no-untyped-def]
siteloom/reindex.py:127: error: Function is missing a type annotation for one or more parameters  [no-untyped-def]
Found 3 errors in 2 files (checked 58 source files)
```

### Why mypy runs in the test job, not the lint job

`ignore_missing_imports = true` is needed for cv2/ultralytics/uiprotect, which ship no
stubs. But pydantic and SQLAlchemy *do* ship `py.typed` — run mypy without the project's
dependencies installed and those silently widen to `Any`, so the strict seed would be
checking a different program and passing for the wrong reason. So mypy runs where the
environment is real.
