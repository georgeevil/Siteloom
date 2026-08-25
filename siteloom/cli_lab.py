"""`siteloom lab` — offline experiments over recorded events.

The lab bootstraps like `jobs list`, not like `run`: config + DB only
(`_light_setup`), never the dispatcher and never the shared vector
store, so it runs alongside a live serve/ingest process. Its sandboxes
are temp directories; the one live-DB write it makes is the
`OperationRun` heartbeat row every CLI job writes, which is what makes
a long embedding pass visible on /jobs and cancellable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from siteloom.cli_library import (
    CONFIG_OPT,
    INTERRUPTED_EXIT,
    QUIET_OPT,
    _light_setup,
    _resume_command,
)

lab_app = typer.Typer(help="Replay recorded events under different settings.")


def _parse_event_ids(text: str) -> list[int]:
    try:
        ids = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError:
        raise typer.BadParameter(f"event ids must be integers: {text!r}")
    if not ids:
        raise typer.BadParameter("at least one event id is required")
    return list(dict.fromkeys(ids))


def _load_variants(path: str | None) -> dict[str, dict]:
    """Named variants from YAML: {name: {dotted.key: value, ...}}.

    A variant may carry `_face_quality: yunet` to flip the lab-level
    quality source alongside its config overrides.
    """
    if not path:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(v, dict) for v in data.values()
    ):
        raise typer.BadParameter(
            f"{path}: expected a mapping of variant name -> {{key: value}}"
        )
    return data


def _event_line(eid: int, scored: dict) -> str:
    ev = scored["events"][eid]
    judged = ev["confirmed_links"] + ev["wrong_links"] + ev["missed_confirmed"]
    tag = " · judged" if judged else ""
    return (
        f"event {eid} ({ev['class']}, {ev['camera']}, "
        f"{ev['frames']} frames{tag})"
    )


def _variant_line(scored: dict, eid: int) -> str:
    ev = scored["events"][eid]
    out = ev["outcomes"]
    return (
        f"  {scored['variant']:<20}"
        f" mints {ev['mints']:>3}  claims {ev['claims']:>3}"
        f"  wrong-links {len(ev['wrong_links']):>3}"
        f"  confirmed {len(ev['confirmed_links'])}"
        f"  pending {out.get('pending', 0) + out.get('mint-budget', 0):>3}"
        f"  ambiguous {out.get('ambiguous', 0):>3}"
    )


@lab_app.command("replay")
def replay(  # noqa: PLR0913 - a lab is knobs
    ctx: typer.Context,
    events: str = typer.Option(..., "--events", help="Comma-separated event ids"),
    config: str = CONFIG_OPT,
    seed: str = typer.Option(
        "reembed",
        "--seed",
        help="Sandbox galleries: reembed (works alongside serve), "
        "copy (exact, needs the store unheld), or none",
    ),
    seed_scope: str = typer.Option(
        "live", "--seed-scope", help="Seed all live identities, or only 'linked'"
    ),
    seed_max_vectors: int = typer.Option(
        20,
        "--seed-max-vectors",
        help="Gallery size to seed per identity — above the live cap is "
        "the 'more vectors' experiment",
    ),
    sets: list[str] = typer.Option(
        None, "--set", help="Override, e.g. person.threshold=0.85 (repeatable)"
    ),
    variants_file: str = typer.Option(
        None, "--variants", help="YAML of named variants: {name: {key: value}}"
    ),
    face_quality: str = typer.Option(
        "detector",
        "--face-quality",
        help="Quality fed to the face identifier's mint/learn gates: "
        "'detector' (YOLO box, live behaviour) or 'yunet' (face score)",
    ),
    include_gated: bool = typer.Option(
        False, "--include-gated", help="Replay frames live ingest gated out too"
    ),
    json_out: str = typer.Option(None, "--json", help="Write the full report here"),
    trace: bool = typer.Option(
        False, "--trace", help="Include per-frame decisions in the JSON report"
    ),
    cache_dir: str = typer.Option(
        None, "--cache-dir", help="Embedding cache (default: .lab-cache beside config)"
    ),
    quiet: bool = QUIET_OPT,
) -> None:
    """Re-resolve recorded events under one or more identity configs.

    Embeds each event's stored crops once (cached), then replays the
    real resolver in a per-variant sandbox seeded from the live
    identities, and scores every variant against the operator's
    verdicts. The live stores are never written.
    """
    from siteloom import lab
    from siteloom.progress import ProgressReporter

    if seed not in ("reembed", "copy", "none"):
        raise typer.BadParameter("--seed must be reembed, copy or none")
    if seed_scope not in ("live", "linked"):
        raise typer.BadParameter("--seed-scope must be live or linked")
    if face_quality not in ("detector", "yunet"):
        raise typer.BadParameter("--face-quality must be detector or yunet")

    event_ids = _parse_event_ids(events)
    named = _load_variants(variants_file)
    site, Session = _light_setup(config)
    cache = Path(cache_dir) if cache_dir else Path(config).parent / ".lab-cache"

    # (name, overrides, face_quality) — baseline first, always.
    runs: list[tuple[str, list[str] | dict, str]] = [("baseline", [], face_quality)]
    if sets:
        runs.append(("tweaked", list(sets), face_quality))
    for name, spec in named.items():
        overrides = {k: v for k, v in spec.items() if not k.startswith("_")}
        fq = spec.get("_face_quality", face_quality)
        runs.append((name, [f"{k}={v}" for k, v in overrides.items()], fq))

    report = None
    with ProgressReporter(
        Session,
        "lab-replay",
        target=f"events {','.join(map(str, event_ids))}",
        resume_command=_resume_command(ctx),
        bar=not quiet,
    ) as progress:
        with Session() as session:
            corpus = lab.build_corpus(session, site, event_ids)
            algo_for = lab.algo_map(site, corpus, [])
            keys = list(algo_for)
            plan = (
                lab.seed_plan(
                    session,
                    keys,
                    scope=seed_scope,
                    event_ids=event_ids,
                    max_vectors=seed_max_vectors,
                    exclude_event_ids=tuple(event_ids),
                )
                if seed != "none"
                else []
            )

            targets = lab.embedding_targets(corpus, plan, site, algo_for)

            with progress.phase("Embedding crops", total=len(targets)):
                bank, embed_stats = lab.embed_corpus(
                    targets, site, cache,
                    tick=progress.advance,
                    check=progress.check_interrupt,
                )
            progress.check_interrupt()

            replayed_crops = frozenset(
                f.crop_path for f in corpus.frames if f.crop_path
            )

            def seeder(sandbox, store):
                if seed == "reembed":
                    return lab.seed_reembed(
                        sandbox, store, plan, bank, algo_for,
                        max_vectors=seed_max_vectors,
                    )
                if seed == "copy":
                    return lab.seed_copy(
                        site.identity.vector_db_path, sandbox, store, plan,
                        max_vectors=seed_max_vectors,
                        exclude_crop_paths=replayed_crops,
                    )
                return {}

            scored: list[dict] = []
            variant_meta: list[dict] = []
            for name, overrides, fq in runs:
                cfg = (
                    lab.apply_overrides(site.identity, list(overrides))
                    if overrides
                    else site.identity.model_copy(deep=True)
                )
                with progress.phase(
                    f"Replay {name}", total=len(corpus.frames)
                ):
                    result = lab.run_variant(
                        name,
                        cfg,
                        corpus,
                        bank,
                        seeder if seed != "none" else None,
                        config=site,
                        algo_for=algo_for,
                        seed_mode=seed,
                        face_quality=fq,
                        include_gated=include_gated,
                        tick=progress.advance,
                        check=progress.check_interrupt,
                    )
                entry = lab.score_variant(result, corpus)
                entry["config"] = lab.config_summary(cfg, keys)
                entry["face_quality"] = fq
                if trace:
                    entry["decisions"] = [vars(d) for d in result.decisions]
                scored.append(entry)
                variant_meta.append({"name": name, "overrides": list(overrides)})
                progress.check_interrupt()

            for entry in scored[1:]:
                entry["vs_baseline"] = lab.compare(scored[0], entry)

            report = {
                "generated_at": datetime.now(timezone.utc)
                .replace(tzinfo=None)
                .isoformat(timespec="seconds"),
                "config": str(config),
                "event_ids": event_ids,
                "seed": {
                    "mode": seed,
                    "scope": seed_scope,
                    "max_vectors": seed_max_vectors,
                    "identities": len(plan),
                },
                "embedding": embed_stats,
                "variants": scored,
                "runs": variant_meta,
            }

    if report is None:
        raise typer.Exit(INTERRUPTED_EXIT)

    baseline = report["variants"][0]
    for eid in event_ids:
        typer.echo(_event_line(eid, baseline))
        for entry in report["variants"]:
            typer.echo(_variant_line(entry, eid))
        typer.echo("")
    for entry in report["variants"][1:]:
        vs = entry["vs_baseline"]
        deltas = ", ".join(
            f"{key} {before}→{after}"
            for key, (before, after) in vs["deltas"].items()
            if before != after
        )
        typer.echo(
            f"{entry['variant']}: {vs['verdict']}"
            + (f" ({deltas})" if deltas else " (no change)")
        )
    if embed_stats["missing_files"]:
        typer.echo(
            f"warning: {embed_stats['missing_files']} crop file(s) missing "
            "on disk — those frames replay as no-embedding"
        )
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2, sort_keys=True))
        typer.echo(f"report -> {json_out}")
