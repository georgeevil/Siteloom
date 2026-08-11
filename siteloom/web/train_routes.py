"""Models: enrolment, face fine-tuning, and what a run may be judged on.

`/training` is *labelling* — crops, names, verify/reject. This screen is
what the system does with those labels, which until now existed only as
five `siteloom train` subcommands with no console at all (CLD-94). The
two were conflated precisely because only one of them had a screen, so
the split is deliberate: "Training data" collects ground truth, "Models"
turns it into vectors and weights.

Four things here are load-bearing and none of them are layout.

**An invalid score is not a zero score.** `EvalMetrics.valid` is False
when a split cannot produce both same-person and different-person pairs;
AUC is undefined there and the dataclass reports 0.0 only because a float
field needs a value. A screen that renders that 0.0 — as a number, a bar,
a percentage — invites adopting a model on no evidence, which is the one
failure `training/face.py` is written to prevent. So the route decides
validity in Python and hands the template either numbers or a reason,
never both, and `adoption_state()` refuses adoption on anything it cannot
stand behind.

**A label without vectors is a name the system cannot see.** The
enrolment backlog is the highest-value half of this page: when the
recognition API returns empty subjects for a person the operator
confirmed last week, unenrolled annotations are the first suspect, and
there was no way to see that outside a terminal. It is shown per person
and runnable here.

**The floor is two numbers, not one** (CLD-95). The Takeout importer
auto-verifies its unambiguous matches with nobody looking, into the same
`verified` column a person's confirm writes. So the readiness panel
reports the per-person floor twice — samples, and samples a human signed
off on — because a fine-tune cleared entirely by Google's people-tags
agreeing with a face detector may be perfectly fine, and is still not the
claim "47 people have five confirmed samples" makes.

**The detector trains detection.** `export-detector`/`detector` improve
face *finding* on your own imagery. Identification stays with the
embedding pipeline, and the panel says so rather than implying a detector
run makes recognition better.

Long runs go through ProgressReporter, so `/jobs` and `siteloom jobs` see
them like any other job, and the vector store is always the process-wide
one — embedded Qdrant allows one client per path per machine.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, or_, select

from siteloom.store import Annotation, Identity, TrainingRun
from siteloom.web import nav, params

log = logging.getLogger(__name__)

#: One model/enrolment job at a time in the serve process. Both jobs
#: write the embedded vector store or hold an embedder, and both are
#: observed through OperationRun like any other long operation — the
#: thread handle is only here to answer "is one already going?".
_train_state: dict = {"thread": None, "kind": None}

#: How often an opaque step tells its OperationRun row it is still alive.
#: Comfortably under the 120 s staleness cutoff.
HEARTBEAT_S = 15.0


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# -- readiness -------------------------------------------------------------


@dataclass
class Readiness:
    """What `train status` shows, plus the gap to a runnable fine-tune."""

    min_samples: int
    samples: int = 0
    people: int = 0
    #: (person, sample_count, clears_the_floor, human_confirmed_count),
    #: commonest first.
    coverage: list[tuple[str, int, bool, int]] = field(default_factory=list)
    ready_people: int = 0
    ready_samples: int = 0
    #: The same floor, counted over human sign-off only (CLD-95). The
    #: importer auto-verifies its pass-1 matches, and those rows are
    #: verified in the same column as the ones a person clicked through —
    #: so "47 people clear the floor" and "47 people have five samples a
    #: human confirmed" are different sentences, and the screen must not
    #: print one while meaning the other.
    human_samples: int = 0
    human_people: int = 0
    human_ready_people: int = 0
    #: {verified_by: count} over the whole set — "unattributed" for rows
    #: verified before the column existed on a database this project
    #: never migrated.
    provenance: dict[str, int] = field(default_factory=dict)
    train_samples: int = 0
    val_samples: int = 0
    can_train: bool = False
    #: Why not, in the operator's terms — empty when it can.
    blockers: list[str] = field(default_factory=list)
    can_evaluate: bool = False
    evaluation_blocker: str = ""
    #: What would have to be verified next, shortest path first.
    gap: list[str] = field(default_factory=list)


def face_readiness(session, min_samples: int, val_fraction: float) -> Readiness:
    """Verified face samples per person, and whether a fine-tune is runnable.

    Deliberately built from `collect_face_samples` and `split_by_person`
    — the same functions the training run itself uses — rather than from
    a count of annotation rows. Only verified, non-rejected annotations
    are training data, an annotation whose source image has gone missing
    is not a sample, and the per-person floor is applied by the collector.
    A page that counted rows would promise a run the trainer then refuses.

    `can_evaluate` is the answer this screen exists to get right: it
    predicts, before anything is trained, whether the validation split can
    produce both same-person and different-person pairs. If it cannot, the
    run's metrics come back `valid=False` — not zero, *unscored* — and no
    result from it may be adopted.
    """
    from siteloom.store import VERIFIED_BY_HUMAN
    from siteloom.training.dataset import (
        collect_face_samples,
        sample_provenance,
        split_by_person,
    )
    from siteloom.training.face import person_coverage

    samples = collect_face_samples(session)
    counts = person_coverage(samples)
    # Filtered from the rows already collected rather than re-queried, so
    # the two readings of the floor can never disagree about their input.
    human_counts = person_coverage(
        [s for s in samples if s.verified_by == VERIFIED_BY_HUMAN]
    )
    readiness = Readiness(
        min_samples=min_samples,
        samples=len(samples),
        people=len(counts),
        coverage=[
            (p, n, n >= min_samples, human_counts.get(p, 0)) for p, n in counts.items()
        ],
        human_samples=sum(human_counts.values()),
        human_people=len(human_counts),
        human_ready_people=sum(1 for n in human_counts.values() if n >= min_samples),
        provenance=sample_provenance(samples),
    )
    ready_names = {p for p, n in counts.items() if n >= min_samples}
    readiness.ready_people = len(ready_names)
    readiness.ready_samples = sum(counts[p] for p in ready_names)

    eligible = [s for s in samples if s.person in ready_names]
    train, val = split_by_person(eligible, val_fraction)
    readiness.train_samples = len(train)
    readiness.val_samples = len(val)

    # The trainer's own floor (training/face.py): two people and four
    # training samples, or it returns without training anything.
    if readiness.ready_people < 2:
        readiness.blockers.append(
            f"a fine-tune needs at least 2 people with {min_samples}+ verified "
            f"samples; {readiness.ready_people} "
            f"{'has' if readiness.ready_people == 1 else 'have'} that many"
        )
    if len(train) < 4:
        readiness.blockers.append(
            f"a fine-tune needs at least 4 verified samples to train on; "
            f"this split has {len(train)}"
        )
    readiness.can_train = not readiness.blockers

    # Verification metrics need both kinds of pair in the held-out half.
    val_counts: dict[str, int] = {}
    for sample in val:
        val_counts[sample.person] = val_counts.get(sample.person, 0) + 1
    same_pairs = any(n >= 2 for n in val_counts.values())
    diff_pairs = len(val_counts) >= 2
    readiness.can_evaluate = same_pairs and diff_pairs
    if not same_pairs:
        readiness.evaluation_blocker = (
            "no person contributes two samples to the validation split, so it "
            "holds no same-person pairs. A run scored on it comes back "
            "unscored — not zero — and cannot be adopted. Verify at least 4 "
            "samples for one person to fix it."
        )
    elif not diff_pairs:
        readiness.evaluation_blocker = (
            "only one person appears in the validation split, so it holds no "
            "different-person pairs. A run scored on it comes back unscored — "
            "not zero — and cannot be adopted."
        )

    short = sorted(
        ((p, min_samples - n) for p, n in counts.items() if n < min_samples),
        key=lambda pair: pair[1],
    )
    if readiness.ready_people < 2:
        need = 2 - readiness.ready_people
        nearest = ", ".join(f"{p} (+{k})" for p, k in short[:3])
        readiness.gap.append(
            f"{need} more {'person' if need == 1 else 'people'} at "
            f"{min_samples}+ verified samples"
            + (f" — closest: {nearest}" if nearest else "")
        )
    if readiness.can_train and not readiness.can_evaluate:
        readiness.gap.append(
            "a person with 4+ verified samples, so the held-out split can be "
            "scored at all"
        )
    return readiness


# -- enrolment backlog -----------------------------------------------------


@dataclass
class Backlog:
    """Verified faces whose embedding never reached the vector store."""

    pending: int = 0
    #: (person, count) for the pending ones — the sweep's actual work.
    people: list[tuple[str, int]] = field(default_factory=list)
    #: Verified but unenrollable as they stand, kept apart from `pending`
    #: for the same reason `failed` is kept apart from `pending` in the
    #: indexer: nothing here will ever be swept in, so folding the two
    #: together would report work as queued that is not.
    no_crop: int = 0
    no_name: int = 0
    enrolled: int = 0
    #: Labelled face identities with an empty gallery — a name the
    #: recognition API cannot return, however many times it was confirmed.
    blind: list[str] = field(default_factory=list)
    blind_total: int = 0


#: Verified, not rejected, not yet swept in. `enroll_verified` walks
#: exactly this set.
def _backlog_filters():
    return (
        Annotation.class_name == "face",
        Annotation.verified.is_(True),
        Annotation.rejected.is_(False),
        Annotation.enrolled.is_(False),
    )


def enrolment_backlog(session, blind_limit: int = 12) -> Backlog:
    """The enrolment gap, per person, plus the labels with no vectors."""
    backlog = Backlog()
    named = func.coalesce(Identity.label, Annotation.proposed_name)
    rows = session.execute(
        select(named, func.count())
        .select_from(Annotation)
        .outerjoin(Identity, Annotation.identity_id == Identity.id)
        .where(
            *_backlog_filters(),
            Annotation.crop_path.is_not(None),
            named.is_not(None),
        )
        .group_by(named)
        .order_by(func.count().desc())
    ).all()
    backlog.people = [(str(name), int(n)) for name, n in rows]
    backlog.pending = sum(n for _, n in backlog.people)

    backlog.no_crop = (
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .where(*_backlog_filters(), Annotation.crop_path.is_(None))
        )
        or 0
    )
    backlog.no_name = (
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .outerjoin(Identity, Annotation.identity_id == Identity.id)
            .where(
                *_backlog_filters(),
                Annotation.crop_path.is_not(None),
                or_(named.is_(None), named == ""),
            )
        )
        or 0
    )
    backlog.enrolled = (
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .where(
                Annotation.class_name == "face",
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
                Annotation.enrolled.is_(True),
            )
        )
        or 0
    )

    blind = session.scalars(
        select(Identity)
        .where(
            Identity.identifier_key == "face",
            Identity.label.is_not(None),
            Identity.vector_count <= 0,
        )
        .order_by(Identity.label)
    ).all()
    backlog.blind_total = len(blind)
    backlog.blind = [i.label for i in blind[:blind_limit]]
    return backlog


# -- past runs and adoption ------------------------------------------------


def adoption_state(run: TrainingRun, config) -> tuple[bool, str]:
    """May this run's artifact be adopted, and if not, why not.

    Every refusal here is the same rule in a different costume: a model is
    adopted on *evidence of held-out improvement*, and nothing else. An
    unscored evaluation is not weak evidence, it is no evidence — which is
    why the invalid case is refused before the improvement flag is even
    read, rather than being allowed through as a 0.0 that lost narrowly.
    """
    if run.kind != "face-embed":
        return False, "only a face-embedding run produces a projection to adopt"
    metrics = json.loads(run.metrics or "{}")
    before = metrics.get("before") or {}
    after = metrics.get("after") or {}
    if not (before.get("valid") and after.get("valid")):
        return False, (
            "the evaluation could not be scored — the validation split had no "
            "same-person (or no different-person) pairs — so this run is "
            "evidence of nothing. An unscored run is not a zero score."
        )
    if not metrics.get("improved"):
        return False, "no improvement on held-out pairs; the base embeddings stand"
    if not run.artifact_path or not Path(run.artifact_path).exists():
        return False, "the projection this run saved is no longer on disk"
    return True, ""


def run_row(run: TrainingRun, config) -> dict:
    """One row of the runs table, with the numbers already gated.

    The template is handed either an evaluation with figures or an
    evaluation with a reason — never a figure it is trusted to hide.
    """
    metrics = json.loads(run.metrics or "{}")
    before = metrics.get("before") or {}
    after = metrics.get("after") or {}
    adoptable, reason = adoption_state(run, config)
    row = {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        "started_at": run.started_at.strftime("%Y-%m-%d %H:%M"),
        "finished_at": (
            run.finished_at.strftime("%Y-%m-%d %H:%M") if run.finished_at else None
        ),
        "samples": run.sample_count,
        "people": run.identity_count,
        "notes": run.notes or "",
        "artifact_path": run.artifact_path,
        "active": bool(
            run.artifact_path
            and config.identity.face_projection_path == run.artifact_path
        ),
        "adoptable": adoptable,
        "not_adoptable_because": reason,
        "evaluation": None,
        "detector_metrics": None,
    }
    if run.kind == "face-embed":
        valid = bool(before.get("valid") and after.get("valid"))
        row["evaluation"] = {
            "valid": valid,
            # Nothing numeric survives an invalid evaluation. `EvalMetrics`
            # fills its floats with 0.0 because a dataclass field needs a
            # value, and 0.0 rendered anywhere on this row would read as a
            # measured result.
            "why": ""
            if valid
            else (
                "not evaluated — the validation split could not produce both "
                "same-person and different-person pairs. Too few people with "
                "2+ samples to form a split; no score was computed, and 0.0 "
                "is not the score."
            ),
            "auc_before": before.get("auc") if valid else None,
            "auc_after": after.get("auc") if valid else None,
            "margin_before": before.get("margin") if valid else None,
            "margin_after": after.get("margin") if valid else None,
            "accuracy_after": after.get("accuracy") if valid else None,
            "threshold": after.get("threshold") if valid else None,
            "best_threshold": after.get("best_threshold") if valid else None,
            "best_accuracy": after.get("best_accuracy") if valid else None,
            "pairs": after.get("pairs") if valid else None,
            "same_pairs": after.get("same_pairs") if valid else None,
        }
    elif metrics:
        # Detector metrics are ultralytics' own (mAP and friends) and
        # measure detection, which the panel says plainly.
        row["detector_metrics"] = {
            str(k): v for k, v in metrics.items() if isinstance(v, (int, float))
        }
    return row


def run_rows(session, config, limit: int = 12) -> list[dict]:
    runs = session.scalars(
        select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)
    ).all()
    return [run_row(r, config) for r in runs]


def detector_readiness(session) -> dict:
    """Verified face boxes available to `train export-detector`."""
    boxes = (
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .where(
                Annotation.class_name == "face",
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
            )
        )
        or 0
    )
    images = (
        session.scalar(
            select(func.count(func.distinct(Annotation.item_id))).where(
                Annotation.class_name == "face",
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
            )
        )
        or 0
    )
    return {"boxes": boxes, "images": images}


def active_projection(config) -> dict:
    """Which face embedding is live, and whether it is really there.

    A configured path that does not exist is the default state of a fresh
    install, and `FaceEmbedder` silently falls back to base SFace — worth
    saying out loud, because "a projection is configured" and "a
    projection is in use" are different facts.
    """
    path = config.identity.face_projection_path or ""
    return {
        "path": path,
        "present": bool(path) and Path(path).exists(),
        "configured": bool(path),
    }


# -- job plumbing ----------------------------------------------------------


def _face_embedder(config):
    """The BASE face embedder, for training only.

    Never loads an existing projection: training starts from the base
    SFace space every time, or repeated fine-tunes stack projections on
    top of each other (`siteloom train face` does the same).
    """
    from siteloom.identity import embedders

    return embedders.build_embedder(
        "face", device=config.detection.device, projection_path=None
    )


def run_with_heartbeat(progress, work, interval: float = HEARTBEAT_S):
    """Run one opaque step while keeping its OperationRun row warm.

    `train_face_projection` takes no progress hook and can spend minutes
    embedding crops without a word. A row whose heartbeat goes cold for
    120 s reports `is_stale`, so a perfectly healthy fine-tune would read
    as abandoned in `/jobs` — the exact inverse of the rule that a dead
    run must never look healthy. The tick advances by **zero**: it claims
    the process is alive, which is all it observes, and never that work
    moved.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = work()
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    worker = threading.Thread(target=target, name="siteloom-train-step", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(interval)
        if worker.is_alive():
            progress.advance(0)
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _save_config(config) -> str | None:
    """Persist the live config to its YAML file, if we know the path."""
    from siteloom.config import save_config

    if not getattr(config, "_source_path", None):
        return None  # built in memory (tests); nothing to write
    try:
        return save_config(config)
    except OSError:
        return None


def register(app, templates, Session, config) -> None:  # noqa: C901 — route table
    nav.add("/train", "Models", "MO", after="/training")

    def ctx(**kw) -> dict:
        return {"site_name": config.site_name or config.site_id, **kw}

    def _face_threshold() -> float:
        ident = config.identity.identifiers.get("face")
        return ident.threshold if ident else 0.36

    def _max_vectors() -> int:
        ident = config.identity.identifiers.get("face")
        return ident.max_vectors_per_identity if ident else 20

    def _busy() -> str | None:
        thread = _train_state["thread"]
        if thread is not None and thread.is_alive():
            return _train_state["kind"]
        return None

    def _refuse_if_busy() -> None:
        """One job at a time, or an actionable 409.

        Both jobs write the one embedded vector store and hold a model,
        so a second concurrent run is not slower, it is wrong. Checked
        before anything else a handler does, so "already running" is
        never reported as some other complaint about the request.
        """
        running = _busy()
        if running:
            raise HTTPException(
                409,
                f"a {running} job is already running in this process — watch it "
                "at /jobs and start the next one when it finishes",
            )

    @app.get("/train")
    def train_page(
        request: Request,
        adopted: int | None = None,
        threshold: float | None = None,
        started: str | None = None,
    ):
        with Session() as session:
            readiness = face_readiness(
                session,
                config.training.min_samples_per_person,
                config.training.val_fraction,
            )
            backlog = enrolment_backlog(session)
            runs = run_rows(session, config)
            detector = detector_readiness(session)
        return templates.TemplateResponse(
            request,
            "train.html",
            ctx(
                readiness=readiness,
                backlog=backlog,
                runs=runs,
                detector=detector,
                projection=active_projection(config),
                face_threshold=_face_threshold(),
                max_vectors=_max_vectors(),
                output_dir=config.training.output_dir,
                detector_model=config.training.detector_model,
                running=_busy(),
                adopted=adopted,
                adopted_threshold=threshold,
                # Only the two banners this page raises itself. Jinja
                # escapes either way, but a note the caller writes is
                # still a note the caller writes.
                started=started if started in ("enrolment", "fine-tune") else None,
            ),
        )

    @app.post("/train/enroll")
    def start_enrolment():
        """Sweep verified-but-unenrolled faces into the live gallery.

        The store is resolved here rather than in the worker so a store
        held by another process answers the operator with an actionable
        503 instead of failing silently inside a background thread.
        """
        from siteloom.web.identity_ops import shared_store

        _refuse_if_busy()
        vectors = shared_store(config, "enrolment sweep")
        max_vectors = _max_vectors()

        def work():
            from siteloom.identity.enroll import enroll_verified
            from siteloom.progress import ProgressReporter
            from siteloom.web.identity_ops import identifier_embedder

            try:
                # The ACTIVE embedder, projection included: an enrolled
                # vector has to land in the same space live matching uses.
                embedder = identifier_embedder(config, "face", app.state.embedders)
                with Session() as session:
                    with ProgressReporter(
                        Session,
                        "face-enroll",
                        target="verified faces awaiting vectors",
                        bar=False,
                        resume_command="siteloom train enroll",
                    ) as progress:
                        with progress.phase("Enrolling verified faces"):
                            enroll_verified(
                                session,
                                vectors,
                                embedder,
                                max_vectors,
                                progress=progress,
                            )
            except Exception:  # pragma: no cover — surfaced via OperationRun
                log.exception("enrolment sweep failed")
            finally:
                _train_state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-enroll", daemon=True)
        _train_state.update({"thread": thread, "kind": "enrolment"})
        thread.start()
        return RedirectResponse("/train?started=enrolment", status_code=303)

    @app.post("/train/face")
    def start_face_training():
        """Fine-tune the face embedding on verified samples.

        Refused up front when the data cannot support a run: starting a
        job that the trainer will decline is worse than saying why.
        """
        _refuse_if_busy()
        with Session() as session:
            readiness = face_readiness(
                session,
                config.training.min_samples_per_person,
                config.training.val_fraction,
            )
        if not readiness.can_train:
            raise HTTPException(400, "; ".join(readiness.blockers))
        threshold = _face_threshold()

        def work():
            from siteloom.progress import ProgressReporter
            from siteloom.training import face as face_mod
            from siteloom.training.dataset import (
                collect_face_samples,
                describe_provenance,
                split_by_person,
            )

            run_id = None
            try:
                with Session() as session:
                    samples = collect_face_samples(
                        session,
                        min_per_person=config.training.min_samples_per_person,
                    )
                    train, val = split_by_person(
                        samples, config.training.val_fraction
                    )
                    run = TrainingRun(
                        kind="face-embed",
                        started_at=_now(),
                        sample_count=len(samples),
                        identity_count=len({s.person for s in samples}),
                    )
                    session.add(run)
                    session.commit()
                    run_id = run.id

                embedder = _face_embedder(config)
                with ProgressReporter(
                    Session,
                    "face-embed",
                    target=(
                        f"{len(train)} train / {len(val)} val verified samples"
                    ),
                    bar=False,
                    resume_command="siteloom train face",
                ) as progress:
                    with progress.phase(
                        "Fine-tuning the face embedding", total=len(samples)
                    ):
                        result = run_with_heartbeat(
                            progress,
                            lambda: face_mod.train_face_projection(
                                train,
                                val,
                                embedder,
                                output_dir=config.training.output_dir,
                                output_dim=config.training.embed_output_dim,
                                epochs=config.training.embed_epochs,
                                lr=config.training.embed_lr,
                                threshold=threshold,
                            ),
                        )
                        progress.advance(
                            len(samples),
                            people=result.people,
                            adopted=1 if result.improved else 0,
                        )
                with Session() as session:
                    row = session.get(TrainingRun, run_id)
                    row.finished_at = _now()
                    row.status = "complete" if result.improved else "no-improvement"
                    row.metrics = json.dumps(result.as_dict())
                    row.artifact_path = result.projection_path
                    # The run states what it trained on, not only how it
                    # scored — see the CLI's `train face`.
                    row.notes = (
                        f"{result.message}\ntrained on: {describe_provenance(train)}"
                    )
                    session.commit()
            except Exception as exc:  # pragma: no cover — via OperationRun
                log.exception("face fine-tune failed")
                # A row left saying "running" outlives the process that
                # was running it; close it out with what went wrong.
                if run_id is not None:
                    with Session() as session:
                        row = session.get(TrainingRun, run_id)
                        if row is not None and row.finished_at is None:
                            row.finished_at = _now()
                            row.status = "failed"
                            row.notes = f"{type(exc).__name__}: {exc}"
                            session.commit()
            finally:
                _train_state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-face-train", daemon=True)
        _train_state.update({"thread": thread, "kind": "fine-tune"})
        thread.start()
        return RedirectResponse("/train?started=fine-tune", status_code=303)

    @app.post("/train/adopt/{run_id}")
    def adopt_run(run_id: int, apply_threshold: str = Form("0")):
        """Make a run's projection the live face embedding.

        Adoption is explicit and never implied by a run finishing —
        `adoption_state` is the whole gate, and it refuses anything whose
        evaluation could not be scored.

        `apply_threshold` defaults to off because it arrives from a
        checkbox, and an unchecked checkbox is not submitted at all: with
        a default of "1" the box could be ticked but never unticked, so
        the one choice this form offers was decided for the operator
        either way.
        """
        apply_threshold = params.one_of(
            apply_threshold, "apply_threshold", ("0", "1")
        )
        with Session() as session:
            run = session.get(TrainingRun, run_id)
            if run is None:
                raise HTTPException(404)
            ok, reason = adoption_state(run, config)
            if not ok:
                raise HTTPException(400, f"this run cannot be adopted: {reason}")
            artifact = run.artifact_path
            metrics = json.loads(run.metrics or "{}")

        tuned = None
        after = metrics.get("after") or {}
        ident = config.identity.identifiers.get("face")
        if apply_threshold == "1" and ident and after.get("best_threshold"):
            # A projection reshapes the similarity distribution, so the
            # old cutoff is stale the moment it is adopted: judging the
            # new space at the old threshold throws most of the gain away.
            #
            # Parsed through the same gate the console uses (CLD-61)
            # rather than trusted because it came from our own row: a run
            # whose metrics JSON is damaged would otherwise write a
            # threshold no face can ever clear into site.yaml, silently,
            # and the /classes page that refuses 42 would then display it.
            tuned = round(
                params.as_similarity(
                    after["best_threshold"], "this run's best_threshold"
                ),
                3,
            )
        # Written only once nothing above can refuse: adoption swaps the
        # live embedding space, and doing that while leaving the cutoff
        # behind is the one combination that is worse than neither.
        config.identity.face_projection_path = artifact
        if tuned is not None:
            ident.threshold = tuned
        _save_config(config)
        # The cached embedder was built in the old space.
        app.state.embedders.pop("face", None)
        url = f"/train?adopted={run_id}"
        if tuned is not None:
            url += f"&threshold={tuned}"
        return RedirectResponse(url, status_code=303)
