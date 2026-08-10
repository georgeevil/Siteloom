# Runbook: does `siteloom service` actually supervise anything? (CLD-105)

Scope: `siteloom service install/start/stop/status/logs/uninstall` against a
real launchd on macOS and a real systemd on Linux.

Everything about *what the unit says* is held by tests — `tests/test_service_render.py`
asserts each directive, including the ones deliberately absent, and
`tests/test_service_cli.py` drives every verb through a fake runner. None of
that establishes the thing the feature is for: that the OS reads these files,
starts the process, restarts it when it should, does not restart it when it
should not, and stops it in time for a batch to commit. A unit file that parses
is not a service that works.

Results go in a dated section at the bottom, as numbered findings, the way
[resumability-runbook.md](resumability-runbook.md) does it.

## 0. Setup

```bash
cp config/site.example.yaml site.yaml     # then edit for the host
siteloom doctor --config site.yaml        # must exit 0 before anything below
siteloom service print-unit --config site.yaml | less   # read it once, by eye
```

Note the `service:` block you are testing against — `stop_timeout_s`, `restart`,
`start_at_boot` all show up as directives.

## 1. The round trip

```bash
siteloom service install --config site.yaml
siteloom service status; echo "exit=$?"        # expect 0
curl -fsS localhost:8000/readyz | head -c 200
siteloom jobs list --config site.yaml          # expect a `serve` row, running
siteloom service stop; siteloom service status; echo "exit=$?"   # expect 3
siteloom service uninstall --yes
siteloom service status; echo "exit=$?"        # expect 4
```

**What is being checked**: the exit codes are the LSB ones a monitoring script
would key on, and `serve` appears in `jobs list` — which is the whole point of
giving it an `OperationRun` row.

## 2. The preflight actually gates (Linux only)

Break the config in a way `doctor` catches — an unwritable `media_dir` is the
easy one — then:

```bash
systemctl --user start siteloom-<site>-serve; echo "exit=$?"
systemctl --user status siteloom-<site>-serve
```

Expect a failed start naming `ExecStartPre`, and **no** server process. Then fix
it and confirm it starts. On macOS there is no `ExecStartPre`; confirm instead
that `siteloom service install` itself reports the problem.

## 3. Restart policy, in both directions

```bash
siteloom service start
kill -9 $(curl -fsS localhost:8000/healthz | python3 -c 'import json,sys;print(json.load(sys.stdin)["pid"])')
sleep 15 && siteloom service status; echo "exit=$?"    # expect 0 — it came back
siteloom jobs reap --config site.yaml                  # the killed row is stale
```

Then the direction that matters more, because it is the one a bare
`KeepAlive=true` gets wrong:

```bash
siteloom service stop
sleep 15 && siteloom service status; echo "exit=$?"    # expect 3 — it stayed down
```

And the crashloop brake: point `--config` at a file that fails at start, then
watch it give up after five tries in five minutes rather than spinning.

## 4. A cancel is not a crash

The `SuccessExitStatus=130` case, which is the one directive with no way to
test it short of this:

```bash
siteloom service install --unit run --config site.yaml   # a config with cameras
siteloom jobs list --config site.yaml                    # note the run id
siteloom jobs cancel <id> --config site.yaml
sleep 20 && siteloom service status --unit run; echo "exit=$?"
```

Expect **3**. A `run` unit without `SuccessExitStatus=130` comes back within
`RestartSec`, silently undoing the operator's cancel.

**macOS cannot express this.** Confirm the documented behaviour instead: on
launchd the job *does* come back, which is why the docs say to stop a
service-managed job with `siteloom service stop` rather than `jobs cancel`.

## 5. SIGTERM lands during an in-flight batch

Start a `library index` over a corpus big enough to run for minutes (build one
with `scripts/make_resume_corpus.py`), started *by a unit* rather than by hand,
then stop the service and time it.

Expect: the stop returns within `TimeoutStopSec`, the run is recorded
`interrupted` with a resume command, and rerunning that command skips what was
done. If the stop times out and the supervisor escalates to SIGKILL, the batch
is lost and `TimeoutStopSec` is too low for this deployment's
`library.batch_size` — that is a finding, not a bug in the unit.

## 6. Reboot survival

```bash
# Linux, user scope: without lingering this is expected to FAIL, which is the point
loginctl show-user "$USER" --property=Linger
sudo reboot
```

After the reboot, `siteloom service status` should be 0 with linger enabled and
4-or-3 without. `install` prints the `enable-linger` warning; confirm it appears
when it should and not when it should not. On macOS a LaunchAgent starts at
*login*, not at boot — confirm that, since the flag is named
`--start-at-boot` and the summary claims to say so.

## 7. Collision refusal

```bash
siteloom service install --config site.yaml                     # serve
siteloom service install --unit run --config site.yaml          # same vector_db_path
```

Expect a refusal quoting the one-client-per-path rule, no unit written, and
`siteloom doctor` reporting the same thing if you force it through with
`--force`.

## 8. A hand-hardened unit

The one thing the default deliberately does not do:

```bash
siteloom service print-unit --config site.yaml > /tmp/hardened.service
# change ProtectSystem=full -> strict, keep ReadWritePaths as generated
```

Install it by hand, start it, and exercise every write path: ingest a clip
(crops under `media_dir`), let identity resolve (the Qdrant directory), let the
sqlite WAL roll, and force a face-weights download with an empty models cache.
If the generated `ReadWritePaths` is complete, all of that works; anything that
fails is a missing entry in `_read_write_paths` and belongs back in the code.

Note that `siteloom service status`/`uninstall` will not recognise this file
once you have edited the marker out.

## Findings

_(dated sections go here)_
