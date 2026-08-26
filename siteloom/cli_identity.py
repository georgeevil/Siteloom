"""`siteloom identity` — the vector store's own maintenance (CLD-106).

One command today: `rebuild`, the reset-and-re-embed after a poisoning
change (crop_margin, the face projection, an embedder dimension). The
web path (/train's Vector store panel) covers the store the serving
process holds; this covers everything else — and refuses, with the
standard message, while another process holds the embedded store.
"""

from __future__ import annotations

from pathlib import Path

import typer

from siteloom.cli_library import (
    CONFIG_OPT,
    INTERRUPTED_EXIT,
    _light_setup,
    _resume_command,
)

identity_app = typer.Typer(help="Vector-store maintenance.")


@identity_app.command("rebuild")
def identity_rebuild(
    ctx: typer.Context,
    config: Path = CONFIG_OPT,
    resume: bool = typer.Option(
        False, "--resume", help="continue an interrupted rebuild from its manifest"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="skip the confirmation prompt"
    ),
):
    """Reset and re-embed every identity gallery from its stored crops.

    For after a poisoning change: labels and Identity rows survive;
    vectors are dropped and rebuilt in the new embedding space, and the
    store is stamped. Interruptible and resumable (--resume); while it
    runs, identities read as unenrolled — degraded and honest.
    """
    cfg, Session = _light_setup(config)
    if not cfg.identity.enabled:
        typer.echo("identity is disabled in this config; nothing to rebuild")
        raise typer.Exit(0)

    from siteloom.identity import VectorStore
    from siteloom.identity.rebuild import plan_rebuild, run_rebuild
    from siteloom.progress import ProgressReporter

    try:
        vectors = VectorStore(cfg.identity.vector_db_path)
    except RuntimeError:
        typer.echo(
            f"{cfg.identity.vector_db_path} is held by another process "
            "(serve?). Use the web rebuild on /train while serve runs, or "
            "stop it first.",
            err=True,
        )
        raise typer.Exit(1) from None

    report = None
    try:
        if not resume:
            with Session() as session:
                plan = plan_rebuild(session, vectors, cfg)
            typer.echo(
                f"This will drop {plan.total_points} vectors across "
                f"{plan.identities} identities and re-embed "
                f"{plan.recoverable_points} of them from stored crops; "
                f"{plan.unrecoverable_points} have no crop anywhere and "
                "cannot be recovered. Labels survive; recognition is "
                "degraded until the rebuild finishes."
            )
            if not yes and not typer.confirm("Proceed?"):
                raise typer.Exit(0)
        with ProgressReporter(
            Session,
            "identity-rebuild",
            target=str(cfg.identity.vector_db_path),
            resume_command=_resume_command(ctx) + (
                "" if resume else " --resume"
            ),
        ) as progress:
            report = run_rebuild(
                Session, vectors, cfg, progress=progress, resume=resume
            )
    finally:
        vectors.close()

    if report is None:
        raise typer.Exit(INTERRUPTED_EXIT)
    typer.echo(
        f"re-embedded {report.vectors_written} vectors across "
        f"{report.identities} identities"
        + (f" (resumed past {report.resumed_past})" if report.resumed_past else "")
        + (f"; {report.crops_unreadable} crops unreadable" if report.crops_unreadable else "")
        + (
            f"; {report.unrecoverable_points} old vectors had no crop and are gone"
            if report.unrecoverable_points else ""
        )
    )
