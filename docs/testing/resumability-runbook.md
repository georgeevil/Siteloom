# Runbook: do long jobs actually survive Ctrl-C and a restart? (CLI-11)

Scope: everything that goes through `ProgressReporter` (`siteloom/progress.py`)
— `library index`, `takeout import`, `train enroll` — plus the observation
surfaces those runs are supposed to light up (`siteloom jobs`, `/jobs`).

The claim under test is the one in CLAUDE.md: *"Every long operation must be
resumable — batch commits plus a skip-what's-done query, never a single
transaction over the whole job."* Three things have to hold:

1. **No lost work** — everything committed before the stop is still there.
2. **No repeated work** — the resumed run skips what finished.
3. **No lying dashboard** — a run that stopped, for any reason, never keeps
   reporting as healthy.

Findings from the first pass through this runbook are in
[Results](#results-2026-08-06) at the bottom. F1–F6 and F8 are fixed; F7, F9
and F10 are open.
Day-to-day operation of a running deployment — `doctor`, health endpoints,
service units — is in [operations.md](../operations.md).

## 0. Setup

### Test data

The repo ships two sample files, which is not enough to interrupt anything.
Two corpora work:

**Synthetic** (fast, disposable, no personal data — use this for the mechanics):

```bash
python scripts/make_resume_corpus.py /tmp/siteloom-resume --count 900
```

Derives a corpus from `samples/bus.jpg`: single-face crops and the two-face
original, jittered per copy, plus Takeout-shaped JSON sidecars tagging two
people. Both passes of the Takeout importer get real work to do (pass 1 names
the single-face photos, pass 2 has to match the two-face ones). Budget
**~10 items/s** for `library index` on an M-series Mac at 810×1080, so 900
images ≈ 90 s of runtime — enough to interrupt twice.

**Real** (the honest test, and it is already on this machine): the Google
Photos Takeout at `~/Downloads/Takeout/Google Photos`, 26,035 items already
registered in `Siteloom-data/archive.db` and **all still `pending`** — the
importer registers items and detects faces, it never marks them `indexed`. So
`library index --all` against `archive.yaml` is a multi-hour job over real data
sitting there ready to go. Copy the DB first (see below).

### An isolated config

Never run the destructive scenarios against `archive.db` in place. Copy it:

```bash
mkdir -p /tmp/siteloom-resume/data && cp /Users/george/dev/Siteloom-data/archive.db /tmp/siteloom-resume/data/
sed -e 's|Siteloom-data/archive.db|/tmp/siteloom-resume/data/archive.db|' \
    -e 's|Siteloom-data/vectors|/tmp/siteloom-resume/data/vectors|' \
    archive.yaml > /tmp/siteloom-resume/archive-test.yaml
```

For the synthetic corpus, take `config/site.example.yaml` and repoint
`storage.db_url`, `storage.media_dir` and `identity.vector_db_path` into
`/tmp/siteloom-resume/`.

### Before every run

- **Stop any `siteloom serve`** on the same `vector_db_path`. Embedded Qdrant
  is one client per path per machine; a serve process holds the lock and the
  CLI will fail to open the resolver. `siteloom doctor --config <cfg>` reports
  this as a failed check naming the remedy, so run it first.
- `-q` is safe again (F2): it hides the bar and keeps the heartbeat, the log
  lines and the Ctrl-C handler.
- Keep a second terminal on `siteloom jobs watch --config <cfg>`; observing
  from another process is half of what is being tested.

### Register the corpus

```bash
CFG=/tmp/siteloom-resume/site.yaml
siteloom library add /tmp/siteloom-resume/library --config $CFG --name resume-test
siteloom library scan --config $CFG          # cheap; must report N pending
```

## 1. Scenario A — Ctrl-C a `library index`, resume it

The baseline case from the issue: item-granular, one commit per item.

```bash
siteloom library index --config $CFG --all      # terminal 1
siteloom jobs watch --config $CFG               # terminal 2
```

Ctrl-C terminal 1 at roughly a third of the way in. Then:

```bash
siteloom jobs list --config $CFG                # expect: interrupted + resume line
sqlite3 <db> 'select status, count(*) from library_items group by status'
```

`jobs list` and `jobs watch` read the database only, so they work while the job
holds the vector store. Cancelling from terminal 2 (`siteloom jobs cancel <id>`)
is equivalent to Ctrl-C in terminal 1 and worth testing both ways.

Pass criteria:

- The run row reads `interrupted`, `current` equals the bar's last position,
  counters preserved, `message = stopped by user`.
- `indexed + pending` = the corpus size; nothing in limbo.
- The resume line is printed and is a command you can paste.
- Rerunning it processes **exactly** the remaining items (`starting (K items)`
  where K = pending) and ends `0 still pending`.
- Second Ctrl-C during the "finishing the current batch" window aborts
  immediately (test this separately — it is the documented escape hatch).

## 2. Scenario B — Ctrl-C a `takeout import`, resume it

The interesting case, because resume here is not a status column but three
different skip-what's-done rules, one per phase.

```bash
siteloom takeout import "/tmp/siteloom-resume/takeout" --config $CFG
```

Interrupt separately in each phase — the phases are announced in the bar, so
stop when the label matches:

| Phase | Resume rule | What to check |
|---|---|---|
| `Scanning archive` | none — pure filesystem walk | restart just redoes it; cost only |
| `Reading metadata` | item looked up by path; tags de-duplicated | rerun must not duplicate `item_tags` rows |
| `Detecting faces` (pass 1) | item skipped if it already has any `face` annotation; gallery rebuilt from prior `unambiguous` crops | `resumed=N` counter climbs; total face count after resume == count from an uninterrupted run |
| `Matching names` (pass 2) | annotations with `proposed_name IS NULL` | see F7 — faces that pass 2 legitimately *cannot* name stay NULL, so they are reprocessed on every resume |

Pass criteria: the union of (interrupted run + resume) produces the same
`annotations` rows — same count, same `proposed_name`/`proposal_basis`
distribution — as one uninterrupted import of the same corpus. Run the
uninterrupted baseline first on a copy of the DB so you have something to
diff against:

```sql
select proposal_basis, verified, count(*) from annotations
where class_name='face' group by 1,2;
```

## 3. Scenario C — kill -9 and machine restart

Ctrl-C is the easy path; the issue title says *restart*.

```bash
siteloom library index --config $CFG --all &
sleep 20; kill -9 %1                    # or: kill -TERM, or close the terminal
siteloom jobs list --config $CFG
```

Pass criteria:

- Committed items survive; the DB is not corrupt or half-written.
- Rerunning the same command completes the remainder.
- `siteloom jobs` must not present a dead run as healthy; `siteloom jobs reap`
  closes it out as `abandoned` with its resume command intact.
- Repeat with `kill -TERM` (what `systemctl stop`, a logout, or a reboot sends)
  and with closing the terminal window (SIGHUP): both now stop as gracefully as
  Ctrl-C, so only `kill -9` should leave a row to reap.

## 4. Scenario D — the observation surfaces

With a run in progress:

- `siteloom jobs watch --config $CFG` in a second terminal — position advances.
- `siteloom serve --config $CFG` → `/jobs` and `/api/jobs` — same numbers.
  (Requires a `vector_db_path` not held by the indexing process, so point serve
  at a config whose identity block is disabled, or accept the lock error.)
- Redirected output (`2> run.log`) must still produce a log line every ~15 s.

Watch for `database is locked` under concurrent read+write: the engine is
plain SQLite with no WAL and no `busy_timeout` tuning
(`siteloom/store/db.py:make_engine`). Heartbeat writes are every 2 s and short,
so it should hold, but it is worth recording if it ever bites during a real
multi-hour run.

## 5. Scenario E — the long one

Only after A–D pass. Against a *copy* of the real archive:

```bash
siteloom library index --config /tmp/siteloom-resume/archive-test.yaml --all 2> index.log
```

26k items at ~10/s ≈ 45 min. Interrupt after 10 minutes, resume, let it finish,
and confirm the totals reconcile. This is also the run that will tell you
whether memory is stable over hours and whether SQLite locking survives real
contention — neither is observable on a 900-image corpus.

---

## Results (2026-08-06)

Scenario A executed end to end on a 300-image synthetic corpus (Apple M-series,
`device: mps`, ~10 items/s). Scenarios B–E not yet run. F1–F6 and F8 are fixed
and re-verified against real runs; F7, F9 and F10 are open.

**What works.** Interrupt handling is sound at the layer it claims to cover.
SIGINT at 195/300 finished the in-flight item, committed, recorded
`interrupted` with `current=195, total=300, boxes=455`, and printed the resume
command. The DB showed exactly 195 `indexed` / 105 `pending`. The printed
command then processed exactly the 105 remaining and ended `0 still pending` —
no lost work, no repeated work. `kill -9` left the DB equally consistent.

### F1 — every interrupt ends in an UnboundLocalError traceback *(fixed)*

`library index`, `takeout import` and `train enroll` all assign their result
*inside* the `with ProgressReporter(...)` block and use it *after*; the reporter
swallows `Interrupted` (`progress.py:__exit__` returns True), so control resumes
after the block with the variable never bound. Observed:

```
Progress saved. Resume with:
  siteloom library index --config …/site.yaml --all
UnboundLocalError: cannot access local variable 'result' where it is not associated with a value
```

Exit code 1. Cosmetic in that nothing is lost — but it undid the reassurance
the resume message had just given, and turned a clean stop into an exit-1 that
any wrapping script reads as failure.

**Fixed**: all three commands pre-bind the variable to `None` and exit with
`INTERRUPTED_EXIT` (130 — 128 + SIGINT, so a wrapper can tell "stopped" from
"crashed") instead of formatting a summary of work that never happened.
Verified end to end: a real Ctrl-C now ends at the resume line with exit 130 and
no traceback. Covered by `tests/test_cli_interrupt.py`.

### F2 — `-q/--quiet` silently disables job tracking and graceful Ctrl-C *(fixed)*

The flag reads "No progress bar (logs only)" but maps to `enabled=False`, which
makes `__enter__` return before creating the `OperationRun` row *and* before
installing the SIGINT handler. Verified: a `-q` run logged normally and added
**no** row — invisible to `siteloom jobs`, `jobs watch` and `/jobs` — and a
Ctrl-C during one would abort mid-batch instead of committing. This is exactly
the failure mode CLAUDE.md's "a dead process must never look healthy" is meant
to prevent, reachable by a flag whose help text promises only to hide a bar.
(`operation_runs` in the real `archive.db` is empty despite a completed 26k
import — consistent with an import run under `-q` or nohup.)
**Fixed**: `ProgressReporter` now takes a separate `bar` flag, and the CLI
passes `bar=not quiet` instead of `enabled=not quiet`. `enabled` stays the
whole-reporter kill switch (tests, embedded callers); hiding the bar keeps the
heartbeat, the log lines and the signal handler. Verified: a `-q` run now
records its `OperationRun` row. Covered by
`test_hiding_the_bar_keeps_the_run_tracked`.

### F3 — a killed job looks healthy for 120 s, and the PID is never used *(fixed)*

After `kill -9`, `siteloom jobs list` showed the run as `running` with
`127/295 43%` and **`eta 16s`** — advertising an ETA for a process that no
longer exists. It flipped to `stale` (with the resume line) only at t+127 s;
`is_stale` needs 120 s of dead heartbeat (`models.py:is_stale`). The row
already stores `pid`; a liveness check (`os.kill(pid, 0)` when the host
matches) would flip it instantly. Smaller bug alongside it: the stale row
still prints `eta 16s`, because `eta_s` gates on the raw `status` column
(still `running`) rather than on `is_stale`.

**Fixed**: `is_stale` now checks the recorded pid on the recording host first
(a new `host` column disambiguates), falling back to the cold heartbeat
elsewhere and after pid reuse; `eta_s` returns None once stale. `siteloom jobs
reap` closes dead rows out as `abandoned`, preserving position and resume
command, and `doctor` warns while any remain.

### F4 — only SIGINT is handled *(fixed)*

`kill -TERM` (a `systemctl stop`, a logout, a shutdown) and SIGHUP (closing the
terminal) kill the process mid-batch: the current batch is lost, the row is
stranded `running`, and no resume line is printed. For a job whose whole point
is surviving a restart, SIGTERM is the signal a restart actually sends.

**Fixed**: `STOP_SIGNALS` covers SIGINT, SIGTERM and SIGHUP; a repeat of the
same signal still aborts immediately. Verified: `kill -TERM` mid-run finished
the batch, committed at 83/228, recorded `interrupted` and printed the resume
command. `siteloom jobs cancel <id>` sends it from another terminal.

### F5 — resume commands are not faithful to the original invocation *(fixed)*

The stored command is rebuilt from two fields, not from the actual arguments:

- `library index` drops `--source-id`, `--limit`, `--no-identify`. Resuming a
  single-source run re-indexes *every* source; resuming a `--no-identify` first
  pass silently turns identification on.
- `takeout import` drops `--limit`, `--batch-size`, `--include-derivatives`,
  `--no-auto-verify`. Resuming a `--no-auto-verify` import starts
  auto-verifying — i.e. writing unreviewed `verified=True` annotations, which
  are training data (`training/dataset.py`).
- Neither quotes the `--config` path.

This is the issue's actual question ("does the printed resume command work?"):
it works, but it can resume a *different* job than the one you stopped.

**Fixed**: the command is rebuilt from the parsed parameters
(`_resume_command(ctx)`), so it cannot drift from the real signature — the
parameters *are* the signature. Non-default flags are all carried, values are
`shlex`-quoted, and `--config` is always explicit so a line read out of `jobs
list` is unambiguous about which deployment it touches. Verified: interrupting
`library index --source-id 1 --no-identify --limit 400` now prints exactly that
back. Covered by three tests in `tests/test_cli_interrupt.py`.

### F6 — the resume line hard-wraps and can't be copy-pasted *(fixed)*

Rich wraps the command at console width; in a non-TTY (80 cols) the path broke
across three lines. **Fixed**: printed with `soft_wrap=True` and no
highlighting — in the job's own terminal and in `siteloom jobs list` — so it
survives a redirect to a log file.

### F7 — pass-2 resume reprocesses every unnamed face

`_pass_two` selects on `proposed_name IS NULL`, but a face it deliberately
leaves unresolved keeps that NULL. Each resume therefore re-embeds every
previously-unresolvable crop. Correct, but the cost grows with each interrupt
on an archive where most faces are untagged. Wants a "pass 2 considered this"
marker.

### F8 — `failed` items are never retried *(fixed)*

`indexer.process` marks a decode failure `status="failed"`, and nothing moved it
back to `pending`; `remaining` counted only `pending`. A resumed run reported
`0 still pending` while items remained unprocessed.

**Fixed**: `ProcessResult` reports `failed_total` separately from `remaining`,
and the CLI prints the retry command whenever failures are outstanding.
`--retry-failed` re-queues them — opt-in, because a corrupt file fails
identically every pass and retrying it on every run over a 26k archive is pure
cost. A new `attempts` column records how many times each file has been tried,
and `library status` lists the worst offenders with their error. Counts are now
scoped to the run's `--source-id`, so a per-source run stops reporting the whole
library's backlog. Verified on three corrupted files: two repaired ones indexed
on retry, the third stayed `failed` at `x2` attempts.

### F9 — `backfill` and `run` are not jobs at all

`siteloom backfill` over a media archive is unambiguously a long-running
operation (PRD §6.6), and `siteloom/cli.py:backfill` uses no `ProgressReporter`:
no `OperationRun` row, no bar, no Ctrl-C handling, no resume, and
`IngestService.run_camera` has no skip-what's-done query. Same for
`siteloom run`. Either they come under the reporter or CLAUDE.md's "any
operation that can run for minutes must go through `ProgressReporter`" is not
true of the codebase. Scoping call for CLI-11: this is the largest gap and
probably its own issue.

### F10 — no automated coverage of any of this

`tests/test_progress.py` sets `interrupt_requested` by hand; nothing delivers a
real signal, exercises a CLI command, or asserts that a resumed run reaches the
same end state as an uninterrupted one. The manual scenarios above are worth
keeping, but they only run when someone remembers to run them.

**Plan.** Three layers, cheapest first. The constraint throughout is
CLAUDE.md's: no model weights, no live cameras — stub modules and synthetic
media only (`tests/conftest.py`).

*Layer 1 — the CLI contract (in-process, milliseconds).* Partly exists in
`tests/test_cli_interrupt.py`: `_setup` is monkeypatched, a stub raises
`Interrupted`, and the test asserts exit 130, an `interrupted` row, and a
faithful resume command. Extend it to deliver a **real** signal —
`os.kill(os.getpid(), SIGINT)` from inside the stub's work loop — and assert the
loop finishes its item, commits, and only then raises. Parametrise over
SIGINT/SIGTERM/SIGHUP so F4 cannot silently regress, and cover the double-signal
abort. Add the same shape for `takeout import` and `train enroll`, whose
post-interrupt paths were the other two F1 sites.

*Layer 2 — the invariant that matters (in-process, ~a second).* **An
interrupted run plus its resume must land exactly where an uninterrupted run
lands.** Nothing tests this today, and it is the whole claim of the feature.
Build it as an equivalence harness over the real `LibraryIndexer` with stub
detection:

1. index a synthetic corpus of N items uninterrupted; snapshot
   `(item.path, status, attempts)` and `(annotation.item_id, bbox, class_name,
   source, verified)` sorted;
2. rebuild the DB, run again with a stub module that sets
   `progress.interrupt_requested = True` after k items, then run the recorded
   resume command;
3. assert the two snapshots are equal, and that no item was processed twice
   (`attempts == 1` everywhere).

No signals, no sleeps, fully deterministic. Parametrise k over a batch boundary,
one item either side of it, and the last item. Then do the same for
`TakeoutImporter.import_tree`, interrupting in each of its three phases — that
is where resume is three different skip-what's-done rules rather than a status
column, and where scenario B's manual work would be replaced.

*Layer 3 — one honest end-to-end (subprocess, seconds; `@pytest.mark.slow`).*
Everything above runs in one process, so it cannot catch a signal that never
reaches a real child, a DB left locked by a killed writer, or a resume command
that is not actually runnable. One test: spawn the real CLI over a synthetic
corpus in a subprocess (stub modules injected via a helper importable from
`tests/`, never an env var baked into production code), **poll the
`OperationRun` row until `current > 0`** rather than sleeping, send SIGTERM,
wait, then assert the row is `interrupted`, run the stored `resume_command`
verbatim through `subprocess.run`, and assert `0 still pending`. A second case
sends SIGKILL and asserts `jobs reap` closes it out. Polling instead of sleeping
is what keeps this from being the flaky test everyone eventually deletes.

Scenarios C (real reboot), D (SQLite contention under real load) and E (the 26k
archive) stay manual — they are about the machine, not the code.

### Test-setup gaps still open

- **B, C, D, E not yet run** — the results above cover scenario A plus the
  kill/SIGTERM probes from C and the `jobs cancel`/`reap` path from D.
- **Real-data pass needs a DB copy and the serve process stopped**; the current
  `archive.yaml` points at live paths and :8324 holds the vector store.
- **Cross-machine restart** (actual reboot, not `kill -9`) is untested; a
  reboot also invalidates the PID check proposed in F3.
- **Concurrent SQLite contention** is untested beyond a single reader.
