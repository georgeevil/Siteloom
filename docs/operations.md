# Operating Siteloom

Running the software is `siteloom serve`. *Operating* it is knowing whether it
is healthy, watching a job you did not start, stopping one cleanly, and clearing
up after a machine that went down. This page covers those.

Two facts explain most of what follows:

- **Embedded Qdrant is one client per path per machine.** `serve`, `frigate`
  and any indexing job all want `identity.vector_db_path`; only one gets it.
  Whatever fails second fails with a Qdrant lock error.
- **Long jobs are resumable, not restartable.** They commit in batches and skip
  what's done, so stopping one is cheap — provided it is stopped with a signal
  it handles rather than killed outright.

## Is this deployment healthy?

```bash
siteloom doctor --config site.yaml     # exit 0 = fit to run, 1 = something failed
siteloom doctor --config site.yaml --json
```

Checks the database and its schema, the media directory (writable, and space
left), the vector store (openable — *and who is holding it*), detector weights,
face weights (by SHA-256, see below), optional plate-OCR dependencies, abandoned
jobs, integration config coherence, and installed service units. Every check
reports a remedy, and one broken check never hides the others.

The service check reads unit *files* and nothing else — no `systemctl`, no
`launchctl`. `doctor` runs as a unit's own `ExecStartPre`, and asking the
service manager about the service it is in the middle of starting is a question
with no good answer and a plausible hang. Live state is
`siteloom service status`, which is never on a boot path.

It is deliberately safe to run at any time, including as an `ExecStartPre` or a
monitoring probe. Exit code 1 means at least one check failed; warnings alone
still exit 0.

## Model weights and their digests

The face pipeline downloads two ONNX files from opencv_zoo on first use and
caches them in `~/.cache/siteloom/models`:

| file | SHA-256 | size |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | 232,589 B |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 38,696,353 B |

Both digests are pinned in `siteloom/identity/embedders.py` and **verified on
every load, cached or freshly downloaded** — not only on the run that fetched
them. The cache is a directory in the operator's home that any local process can
write, and these weights decide who the system thinks a face belongs to, so
"the file is present and roughly the right size" is not the question. A file
that fails is deleted before anything else happens; leaving it would mean the
next run loads it without even trying to re-download.

`siteloom doctor` verifies the cache in place and reports, but never deletes —
it stays safe to run as an `ExecStartPre`.

**A mismatch is one of three things.** A corrupted or interrupted download (a
clean re-download fixes it, and that is what deleting the file achieves); a
cache someone else wrote to; or an upstream that republished the artifact under
the same name. Do not edit the pinned digest to make the error go away — that
turns the check off. Confirm the new file against opencv_zoo's own git-lfs
pointer first, which is an independent source from the blob download:

```bash
curl -sSL https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/\
face_detection_yunet/face_detection_yunet_2023mar.onnx   # prints "oid sha256:…"
```

### Pre-seeding an offline or air-gapped install

A verified cached file is used without touching the network, so a host with no
outbound access works as long as the weights are already there. Copy them from a
machine that has them (or download and check them by hand), then either drop
them in `~/.cache/siteloom/models` or point `SITELOOM_MODELS_DIR` at the
directory holding them:

```bash
mkdir -p /opt/siteloom/models && cp face_*_2021dec.onnx face_*_2023mar.onnx /opt/siteloom/models
shasum -a 256 /opt/siteloom/models/*.onnx        # compare with the table above
export SITELOOM_MODELS_DIR=/opt/siteloom/models  # set it in the service unit too
siteloom doctor --config site.yaml               # "face models  2/2 verified in …"
```

The env var applies to `serve`, `frigate`, and every CLI job; set it in the
launchd plist / systemd unit as well as your shell, or a service will fall back
to the home cache and try to download.

The YOLO detector weights are a separate story: ultralytics downloads
`detection.model` itself, inside its own library, and Siteloom never sees that
transfer — so it is **not** covered by any of the above. Point
`detection.model` at a local path you vetted if that matters to you.

## Watching a running server

```
GET /healthz   -> 200 {"status": "ok"}
GET /readyz    -> 200 or 503 {"ok": bool, "checks": [{"name": ..., "status": "ok|warn|fail"}]}
```

`/healthz` is liveness: it touches nothing but the process, so a slow database
cannot get the server killed and restarted into the same slow database.
`/readyz` is readiness: it runs the cheap, read-only half of `doctor` (database,
media dir, jobs) and returns 503 when the process is up but cannot do its job.
It never opens the vector store — this process is already holding it — and it
never migrates: schema work belongs to `init-db` and to `run`/`serve` startup.

Both endpoints are public (a probe that needs a cookie gets the service killed),
so their bodies are deliberately terse: check names and statuses, never paths,
hostnames, pids or error text — that detail, with remedies, is `siteloom
doctor` at the terminal. `/readyz` caches its answer for ~5 s, so hammering it
costs one check run per window, not one per request.

`/jobs` in the web UI and `GET /api/jobs` show the same job rows as the CLI.

## Exposing the console

Accounts (`siteloom users add`) and sign-in throttling are built in, and the
session cookie hardens itself: `HttpOnly` and `SameSite=Lax` always, `Secure`
automatically whenever the request arrived over HTTPS — directly or via a
reverse proxy or tunnel that sets `X-Forwarded-Proto: https`, which Caddy,
nginx's standard proxy snippets, and Cloudflare/Tailscale tunnels all do.
Cross-site form posts are refused by an `Origin`/`Sec-Fetch-Site` check in the
same middleware that authenticates, so there is nothing to configure — just
keep the proxy passing the browser's `Host` header through (or setting
`X-Forwarded-Host`), because the check compares the browser's origin against
it. Sessions last 14 days from sign-in with no idle timeout (a wall console
should not log itself out mid-shift); sign-out, disabling the account, or
revoking sessions on `/users` ends one sooner, and expired rows are pruned at
each sign-in. Serving the console over TLS is the proxy's job — Siteloom
itself speaks plain HTTP.

## What clock the console shows (CLD-100)

Storage is naive UTC by contract; the console converts at display, using one
per-site timezone, so every viewer sees the cameras' wall clock — two
operators discussing "the 21:40 event" mean the same moment, and an incident
export names the zone its times are in. The setting is `timezone:` in
site.yaml (an IANA name, e.g. `Europe/Bucharest`), managed from the **Site
time** panel on `/classes` (admin): type or pick a zone, click **Detect from
NVR** to adopt what the UniFi NVR is configured to (one connect-read-
disconnect), or — while nothing is set — accept the zone the browser proposes
via the Intl API (no geolocation, no permission prompt). The panel names
which of those supplied the current value. Unset means UTC, labelled as UTC.
Times typed into console forms (the backfill range, manual bookings) are read
in the same zone; the server's own OS timezone never matters.

## Watching and steering jobs

Every long operation heartbeats a row, so any terminal can see it:

```bash
siteloom jobs list  --config site.yaml     # recent runs, outcomes, resume commands
siteloom jobs watch --config site.yaml     # live view of what's running now
```

These read the database only — no vector store, no models — so they work *while*
a job is running, which is the only time you need them.

Stopping a job you did not start:

```bash
siteloom jobs cancel 12 --config site.yaml            # graceful: finishes the batch
siteloom jobs cancel 12 --config site.yaml --force    # SIGKILL; loses the batch
```

Graceful cancel sends the same signal Ctrl-C does. The job finishes its current
batch, commits, records itself `interrupted`, and prints its resume command in
its own terminal. `cancel` refuses runs belonging to another host, and tells you
to reap instead when the process is already gone.

Clearing up after a crash or a reboot:

```bash
siteloom jobs reap --config site.yaml     # mark dead runs `abandoned`
```

A process killed with `kill -9`, or lost to a power cut, leaves its row saying
`running` forever. Reaping closes those out while preserving their position and
resume command, so `jobs list` stops being ambiguous. Runs whose process is
provably gone are detected immediately by pid; on another host, or after pid
reuse, a two-minute cold heartbeat is the backstop.

## The daily labeling queue (CLD-8)

`/training` opens with **Today's queue**: roughly twenty borderline judgments,
keyboard-workable (`J`/`K` move, `Y` confirm, `N` reject), sized to clear in
about ten minutes. That ten minutes is the label-and-learn habit — the queue
picks the crops where one label moves the model most, per Frigate's guidance
(cited in `docs/identity-management-analysis.md`): label the clear borderline
crops, not the 90%-confident ones.

What "borderline" means, in trust order: unreviewed identity matches whose
similarity sits close to their identifier's threshold (plate matches are
excluded — their similarity is synthetic); unconfirmed name proposals from the
Takeout importer; and unnamed crops whose detection confidence sits in the
middle of the range. `Annotation` stores no match similarity, so proposals are
queued as a tier rather than ranked by nearness to the face threshold.

The mechanics protect the habit:

* **Deterministic within a day.** The order is seeded by the date — reloading
  never reshuffles mid-session.
* **Judged items leave and nothing slides in.** The session converges to zero;
  tomorrow brings a fresh rotation through the borderline region. Crops indexed
  today wait for tomorrow's queue.
* **"Nothing borderline today" is a good state.** It means no matches near a
  threshold and no proposals waiting — not a broken screen.
* **It works while ingest runs.** Selection is SQL only; the queue never opens
  the vector store. Confirming a face still enrolls its embedding, so *that*
  action answers 503 while another process holds the store — rejects and
  verdicts go through regardless.

Every action posts to the console's existing review endpoints, so verification
provenance (who confirmed, when) is stamped in the one place it always was.

## Stopping things

| Signal | Sent by | Effect |
|---|---|---|
| SIGINT | Ctrl-C, `jobs cancel` | finish batch, commit, record `interrupted`, print resume command |
| SIGTERM | `siteloom service stop`, shutdown | same as SIGINT |
| SIGHUP | closing the terminal | same as SIGINT |
| the same signal twice | an impatient operator | immediate abort; the batch in flight is lost |
| SIGKILL | `kill -9` | immediate death; committed work survives, the row needs reaping |

`serve` follows the same table. uvicorn owns the live shutdown, but the signal
is recorded underneath it, so a `systemctl stop` or a `jobs cancel` leaves the
run marked `interrupted` with a resume command and exits 130 — rather than
dying where the row still says `running` and only a reap can close it out.

Give a job **at least a batch's worth of time** to stop. Batch size is
`library.batch_size` (default 100) for indexing and `--batch-size` (default 200)
for Takeout imports, so a 30-second stop timeout is usually generous and a
5-second one is not.

## Running as a service

```bash
siteloom service install --unit serve --config site.yaml
siteloom service status                # 0 running, 3 stopped, 4 not installed
siteloom service stop | start | restart
siteloom service logs -f
siteloom service uninstall
```

One verb set over both supervisors: a LaunchAgent plist on macOS
(`launchctl bootstrap`/`bootout`/`kickstart` — the modern calls, not the
deprecated `load`/`unload`), a unit file on Linux (`systemctl --user`).
`--unit` selects `serve` (default), `run`, or `frigate`; `--scope system` writes
a system unit instead of a per-user one.

Siteloom does not daemonize itself. There is no `--daemon` and no PID file: the
process stays in the foreground, logs where it is told, and stops on SIGTERM,
and the supervisor owns backgrounding, restart and boot ordering. The liveness
answer is `/healthz` and the `OperationRun` row, both of which know more than a
pid file could. On a box with no service manager at all, run `siteloom serve`
under whatever supervises things there — the unit `print-unit` renders is a
reasonable starting point.

### Review before you install

```bash
siteloom service print-unit --unit serve --config site.yaml
```

Renders exactly what `install` would write, and writes nothing. It works on any
platform, including ones with neither supervisor, because rendering is pure —
so it is also the way to hand a unit to config management or to hand-edit one.
The catch: `status` and `uninstall` recognise a unit by the marker `install`
puts in it (`X-Siteloom-Generator`, or `SITELOOM_SERVICE_UNIT` in the plist's
environment), so a hand-edited copy that loses the marker becomes yours to
manage. `install` and `uninstall` both refuse to touch an unmarked file rather
than clobber somebody's deliberate work; `--force` overrides.

Reinstalling after a config change shows a diff and asks. Changing
`service.port` alone produces *no* diff, and that is the design: the unit reads
the config rather than copying it, so there is one place to change a port.

### What the generated unit says, and why

Everything below comes from the `service:` section of the config, so copying
`site.yaml` to another host copies the deployment's shape with it.

| Directive | Value | Reason |
|---|---|---|
| `ExecStart` | absolute program, absolute `--config` | a bare `siteloom` resolves against systemd's minimal PATH — the classic "works in my shell" failure |
| `WorkingDirectory` | the config file's own directory | `_ANCHORED_PATHS` makes that the one place every relative path in the YAML resolves correctly, and `storage.db_url` is deliberately *not* anchored, so a relative `sqlite:///` follows it |
| `ExecStartPre` | `siteloom doctor --config …` | `doctor` is safe to run at any time and exits non-zero only on a real failure, so it gates a boot into a broken deployment. launchd has no equivalent: on macOS the gate runs at install time instead |
| `Type=exec` | not `simple` | `simple` calls a start successful before the exec is attempted, so a moved venv looks like a healthy service that is not there. `--notify` opts into `Type=notify` (see below) |
| `Restart=on-failure`, `RestartSec`, `StartLimitBurst=5` / `StartLimitIntervalSec=300` | | without a brake a config error respawns forever at full speed. launchd: `KeepAlive={SuccessfulExit: false}` and `ThrottleInterval` — a bare `KeepAlive=true` also resurrects a server you stopped on purpose |
| `SuccessExitStatus=130` | every unit | a stopped process exits 130 by design — `serve` included, since it records the stop and then says so. Without this, `Restart=on-failure` reads a deliberate stop as a crash and brings it straight back. **launchd cannot express it** — on macOS, stop a service-managed process with `siteloom service stop`, not `jobs cancel` |
| `TimeoutStopSec` | 30 s for `serve`, 60 s for `run`/`frigate` | a batch's worth of time; see the stop-signal table above |
| `KillSignal`, `KillMode` | *absent* | the defaults (SIGTERM, control-group) are already right. `KillMode=mixed` is the tempting wrong answer: ingest's workers are threads, not children |
| `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`, `ReadWritePaths=…` | | `ProtectSystem=strict` fails minutes into a run when one write path is missing, so `full` is the default and the computed `ReadWritePaths` (media dir, vector store, the sqlite *directory* — WAL and SHM are siblings — training output, log dir, models dir) is emitted anyway so tightening it by hand is a one-word edit |
| `ProtectHome` | *absent* | the face weights cache is `~/.cache/siteloom/models` unless `SITELOOM_MODELS_DIR` moves it |
| `RuntimeDirectory`, `StateDirectory`, `LogsDirectory` | *absent* | the config file decides where state lives (CLD-64). Two mechanisms answering that question is the bug those directives would reintroduce |
| `ProcessType` (launchd) | *absent* | `Background` puts the job in a throttled task-policy band that would starve YOLO inference on the Apple Silicon target |
| no `--workers` | | embedded Qdrant is one client per path per machine, so a second uvicorn worker cannot open the vector store. The generated unit says so in a comment |

Logging: every unit passes `--log-file`, so the project's own rotating handler
(10 MB × 3) owns the real log. launchd does not rotate `StandardOutPath`, so the
plist's streams stay a crash channel and nothing else. `siteloom service logs`
knows the difference — `journalctl` on Linux, a `tail` over both files on macOS.

### Two things the CLI reports rather than doing

- **`loginctl enable-linger`.** A `--user` unit stops at logout and does not
  start at boot without it. `install` checks and prints the exact
  `sudo loginctl enable-linger <you>` command; it never runs `sudo` for you.
- **`--scope system` on macOS.** Writing to `/Library/LaunchDaemons` needs
  root, so `install` renders the plist and prints the two `sudo` commands
  instead of running them.

### `Type=notify` is opt-in

`siteloom service install --notify` emits `Type=notify` and `serve` sends
`READY=1` once the sockets are actually bound, so `systemctl start` blocks until
the server can answer rather than returning as soon as the process exists.

It is not the default because `Type=exec` already catches the failure that
actually happens — a bad exec — and a readiness signal has to be told the truth
by hand. `WatchdogSec` is deliberately not offered at all: the only place to
ping from is a background thread, and a thread keeps pinging happily while the
event loop is wedged. A watchdog that lies is worse than none; `/readyz` is a
better liveness probe because it goes through the loop.

### One vector store, one process

`serve`, `frigate` and every indexing/enrolling job open the embedded Qdrant
directory exclusively. Options, in order of preference:

1. Run the long job when the server is stopped (`siteloom service stop`), which
   is what `doctor` will tell you to do.
2. Give the job a config whose `identity.enabled` is false, when it does not
   need identification (`library index --no-identify`).
3. Move to a real Qdrant server for `identity.vector_db_path` — the same client
   class speaks to both, so this is a config change (and the V1 multi-site
   direction anyway).

`siteloom doctor` reports exactly this failure with the remedy, so "why won't my
job start?" is one command rather than a stack trace.

Two units that would collide are now caught earlier than that: `service install`
refuses to write a second unit whose config names the same
`identity.vector_db_path` as one already installed (`--force` if you are about
to change one of them), and `doctor`'s `services` check reports the clash
between installed units. Both quote the same remedy. A restart at 4am is a late
time to discover a configuration that could never have worked.

## Starting over (`siteloom reset`)

```bash
siteloom reset --config site.yaml --dry-run   # inventory only, changes nothing
siteloom reset --config site.yaml             # prints the same, then asks
siteloom reset --config site.yaml --yes       # no prompt (scripts)
```

Erases everything the site has observed and learned: events, detections,
identities and their vectors, plate reads, incidents, the library index and its
crops. What survives is what was never learned in the first place — the config
(cameras, credentials, thresholds), the library's **original archives**,
downloaded model weights, and the logs. `--keep-users` holds back operator
accounts and their sessions.

Three properties are worth knowing before you run it:

- **All three stores or none.** Rows, `media_dir` and the vector directory are
  cleared together. Clearing the rows alone would leave galleries that still
  match faces belonging to identities that no longer exist.
- **It refuses rather than half-works.** A running `serve`/`run`/job (a live
  `OperationRun`, not a stale one) blocks it, because that process holds the
  vector store open and would write fresh rows into the database underneath.
  So does a registered library source that sits *inside* `media_dir` — clearing
  it would delete the operator's own archive, so nothing is removed at all.
- **Ids restart at 1.** A reset install should not hand out event 1260 as its
  first event.

Only the CLI can do this; there is deliberately no console button.

## Logs

`serve`, `frigate` and every job share the project's logging setup: a Rich
console handler on a terminal, plain lines when redirected, and an optional
rotating file (10 MB × 3) with `--log-file`. A job run with output redirected
also emits a progress line every 15 seconds, so a background run is never
silent.
