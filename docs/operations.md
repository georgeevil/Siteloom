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
left), the vector store (openable — *and who is holding it*), detector and face
model weights, optional plate-OCR dependencies, abandoned jobs, and integration
config coherence. Every check reports a remedy, and one broken check never hides
the others.

It is deliberately safe to run at any time, including as an `ExecStartPre` or a
monitoring probe. Exit code 1 means at least one check failed; warnings alone
still exit 0.

## Watching a running server

```
GET /healthz   -> 200 {"status": "ok", "site": ..., "pid": ...}
GET /readyz    -> 200 or 503 {"ok": bool, "checks": [...]}
```

`/healthz` is liveness: it touches nothing but the process, so a slow database
cannot get the server killed and restarted into the same slow database.
`/readyz` is readiness: it runs the cheap half of `doctor` (database, media dir,
jobs) and returns 503 when the process is up but cannot do its job. It never
opens the vector store — this process is already holding it.

`/jobs` in the web UI and `GET /api/jobs` show the same job rows as the CLI.

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

## Stopping things

| Signal | Sent by | Effect |
|---|---|---|
| SIGINT | Ctrl-C, `jobs cancel` | finish batch, commit, record `interrupted`, print resume command |
| SIGTERM | `systemctl stop`, launchd, shutdown | same as SIGINT |
| SIGHUP | closing the terminal | same as SIGINT |
| the same signal twice | an impatient operator | immediate abort; the batch in flight is lost |
| SIGKILL | `kill -9` | immediate death; committed work survives, the row needs reaping |

Give a job **at least a batch's worth of time** to stop. Batch size is
`library.batch_size` (default 100) for indexing and `--batch-size` (default 200)
for Takeout imports, so a 30-second stop timeout is usually generous and a
5-second one is not.

## Running as a service

### macOS (launchd) — the primary target

`~/Library/LaunchAgents/dev.siteloom.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>            <string>dev.siteloom.serve</string>
  <key>WorkingDirectory</key> <string>/Users/you/dev/Siteloom</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/dev/Siteloom/.venv/bin/siteloom</string>
    <string>serve</string>
    <string>--config</string><string>site.yaml</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8000</string>
    <string>--log-file</string><string>/Users/you/dev/Siteloom-data/serve.log</string>
  </array>
  <key>RunAtLoad</key>     <true/>
  <key>KeepAlive</key>     <true/>
  <key>StandardErrorPath</key> <string>/Users/you/dev/Siteloom-data/serve.err</string>
</dict>
</plist>
```

```bash
launchctl load  ~/Library/LaunchAgents/dev.siteloom.serve.plist
launchctl unload ~/Library/LaunchAgents/dev.siteloom.serve.plist   # stops it
```

launchd sends SIGTERM on unload, which `serve` and any job handle gracefully.

### Linux (systemd)

```ini
[Unit]
Description=Siteloom web UI
After=network.target

[Service]
WorkingDirectory=/opt/siteloom
ExecStartPre=/opt/siteloom/.venv/bin/siteloom doctor --config site.yaml
ExecStart=/opt/siteloom/.venv/bin/siteloom serve --config site.yaml --port 8000
Restart=on-failure
# Long enough for an in-flight batch to commit.
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

A second unit for `siteloom frigate` follows the same shape. **Do not** run
`serve` and `frigate` against the same `identity.vector_db_path` — see below.

### One vector store, one process

`serve`, `frigate` and every indexing/enrolling job open the embedded Qdrant
directory exclusively. Options, in order of preference:

1. Run the long job when the server is stopped (`launchctl unload …`,
   `systemctl stop siteloom`), which is what `doctor` will tell you to do.
2. Give the job a config whose `identity.enabled` is false, when it does not
   need identification (`library index --no-identify`).
3. Move to a real Qdrant server for `identity.vector_db_path` — the same client
   class speaks to both, so this is a config change (and the V1 multi-site
   direction anyway).

`siteloom doctor` reports exactly this failure with the remedy, so "why won't my
job start?" is one command rather than a stack trace.

## Logs

`serve`, `frigate` and every job share the project's logging setup: a Rich
console handler on a terminal, plain lines when redirected, and an optional
rotating file (10 MB × 3) with `--log-file`. A job run with output redirected
also emits a progress line every 15 seconds, so a background run is never
silent.
