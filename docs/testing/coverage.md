# Coverage (CLD-71)

Before this, gaps were found by reading. Now `pytest` measures them.

## Running it

```
.venv/bin/pytest                     # suite + terminal coverage report
.venv/bin/pytest --cov-report=html   # …and htmlcov/index.html
.venv/bin/pytest --no-cov tests/x.py # fast loop, no measurement
```

`--cov` is in `addopts`, so there is no separate coverage command to forget. The cost is
real and worth knowing — measured on one clean venv, same machine, same 620 tests:

| | Full suite | `tests/test_zones.py` alone |
| --- | --- | --- |
| `--no-cov` | 1m52s | 0.20s |
| with coverage | 3m57s | 2.4s |

Branch coverage roughly doubles wall clock; on a single small file the fixed cost of
reporting over 58 modules dominates outright. Use `--no-cov` while iterating on one test.

`COVERAGE_CORE=sysmon` (coverage's `sys.monitoring` backend, 3.12+) brings the full suite
down to **3m06s** for the same 76%. It is not the default here because that run differed
by one partial branch (253 vs 254) and raised an extra warning — a measurement backend
that disagrees with itself is not what you want under a floor. Worth revisiting.

## What is measured, and what is not

`source = ["siteloom"]` — the shipped package.

* `tests/` is out because a test file's own coverage is tautological.
* `scripts/` is out because it is operator tooling outside the wheel. This one is a
  genuine loose end rather than a principle: `scripts/archive_report.py` (206 statements)
  *has* tests in `tests/test_archive_report.py` and is simply unmeasured. Adding it is a
  one-word change to `source`.

**Nothing is omitted at module granularity.** In particular `adapters/unifi.py` (37%),
`identity/embedders.py` (22%) and `web/live.py` (78%) stay in. Omitting a module is how
a coverage number becomes a lie — it deletes the denominator along with the gap. Those
files score low because CLAUDE.md forbids tests that reach real cameras or model
weights, which is exactly the fact the number should be reporting.

Line exclusions (`exclude_also`) are limited to constructs with no reachable body:
`if TYPE_CHECKING:`, `@overload`, `@abstractmethod`, `raise NotImplementedError`,
`if __name__ == "__main__":`, `Protocol` bodies. There is no "hard to test" pragma.

`branch = true`, because this codebase is mostly guard clauses — optional-import
fallbacks, threshold/margin gates, `if verified and not rejected`. A statement-only
percentage over that shape counts the `if` and says nothing about the branch never taken.

## Baseline

Measured on a clean Python 3.12 venv built from `requirements-dev.txt`, 620 tests passing:

```
TOTAL   7180 stmts   1522 missed   1982 branches   254 partial   76%
```

13 modules are at 100%. The floor in CI is **70%** — six points of slack, so an honest
refactor cannot redden `main` but an untested new module still does.

### The floor is in the workflow, not in the config

`fail_under` is set as `--cov-fail-under=70` on the CI `pytest` step, deliberately not in
`[tool.coverage.report]`. Put it in the config and `pytest tests/test_zones.py` — the
single-test command CLAUDE.md documents — exits 1 with "Required test coverage of 70.0%
not reached. Total coverage: 2.69%". A floor asserted over a subset is asserting
something untrue.

## Where the gaps are

Ordered by how much untested code each represents, not by percentage:

| Module | Cover | Missed | Why, and is it worth closing |
| --- | --- | --- | --- |
| `cli_library.py` | 25% | 365 | The library/takeout/classes/train/jobs CLI surface. Largest single gap. Worth closing: these are the commands operators actually run. |
| `cli.py` | 32% | 149 | Same story for the root CLI. `tests/test_cli_interrupt.py` covers only the interrupt path. |
| `web/library_routes.py` | 78% | 132 | Well covered in absolute terms; the misses cluster in the review/proposal handlers. |
| `training/face.py` | 36% | 94 | Fine-tune + eval loop. Needs torch and real images — the honest answer is a slow marker, not a stub. |
| `ingest.py` | 79% | 71 | Misses are the live-camera and audio branches. |
| `identity/embedders.py` | 22% | 70 | Needs YuNet/SFace weights. Structurally untestable under the current contract. |
| `adapters/unifi.py` | 37% | 58 | Needs an NVR. Same. |
| `cli_users.py` | 19% | 54 | **Cheap to close** — pure CLI over the `User` table, no hardware, no weights. |
| `library/indexer.py` | 75% | 41 | |
| `progress.py` | 84% | 39 | |
| `identity/plates.py` | **0%** | 37 | Optional `requirements-plates.txt` is not installed in CI, so the module never imports. A `pytest.importorskip` suite would move this off zero. |
| `modules/identity.py` | 23% | 25 | Embedding compute; blocked by the same weights as `embedders.py`. |
| `modules/audio.py` | 56% | 26 | `detect_episodes` is the pure part and is tested; the file-decode wrapper is not. |

## Ratchet plan

1. **70 today.** Anti-collapse only.
2. **→ 78** after `cli_users.py` and `identity/plates.py` (an importorskip suite). Both
   are hardware-free and together are ~90 statements.
3. **→ 82** after the `cli_library.py` command surface gets Typer-runner tests.
4. Stop there. Pushing past ~85 means testing `embedders.py`, `unifi.py` and
   `training/face.py`, which means weights and cameras in CI. That trade is worse than
   the number.

Each step is an edit to `--cov-fail-under` in `.github/workflows/ci.yml`, done on the PR
that earns it. The floor never reads HEAD automatically — a ratchet that sets itself is
a ratchet that locks in a bad afternoon.

## CI cost

Measured on a clean 3.12 venv:

* `pip install -r requirements-dev.txt`: **6m34s**, **5.4 GB** installed. The linux pins
  pull the full CUDA wheel stack (`nvidia-*`, `torch`, `triton`) even though nothing in
  the suite imports torch — `siteloom/identity/embedders.py` and `siteloom/training/face.py`
  import it lazily, inside functions.
* Suite: ~4 min with coverage.

So the `test` job is roughly 11 minutes cold, ~5 warm with the pip cache. The obvious
follow-up is a CPU-only torch install for CI, which would cut several GB and most of the
install time — but it means either a second constraints file or re-pinning
`requirements.txt` off the CUDA wheels, and that changes what Linux developers get. Left
as a deliberate follow-up rather than smuggled in here.
