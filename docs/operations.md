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
jobs, and integration config coherence. Every check reports a remedy, and one broken check never hides
the others.

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
