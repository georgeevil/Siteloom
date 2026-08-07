"""Fine-tuning the face embedding for *your* people.

SFace is trained on a generic face corpus. Its embedding space separates
arbitrary strangers well, but it knows nothing about which distinctions
matter in your archive. Fine-tuning here means learning a linear
projection W (128 -> D) applied after SFace, trained so that embeddings
of the same person move together and different people move apart.

Why a linear head rather than fine-tuning the backbone: with a few
hundred to a few thousand face crops from a personal archive, updating
28M backbone weights overfits immediately, and the OpenCV SFace model is
a frozen ONNX graph anyway. A projection trained with a proxy/NCA-style
loss is the standard cheap win — it typically buys a solid margin
improvement on the people you actually care about, trains in seconds on
CPU, and drops straight into FaceEmbedder as a matrix multiply.

Evaluation is verification-style (are two crops the same person?), which
is what the identity resolver actually does at runtime, reported as ROC
AUC plus accuracy at the configured threshold.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from siteloom.training.dataset import FaceSample

log = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    auc: float = 0.0
    accuracy: float = 0.0
    threshold: float = 0.0
    same_mean: float = 0.0
    diff_mean: float = 0.0
    margin: float = 0.0
    pairs: int = 0
    same_pairs: int = 0
    # Threshold that maximizes accuracy on this data, and the accuracy it
    # achieves. A projection changes the similarity distribution, so the
    # previously-tuned operating point is stale afterwards — reporting
    # only accuracy at the old threshold hides most of the improvement.
    best_threshold: float = 0.0
    best_accuracy: float = 0.0
    # False when the set cannot produce a meaningful score — fewer than
    # two samples, or no same-person (or no different-person) pairs. An
    # invalid score is NOT a zero score; callers must not compare it.
    valid: bool = False

    def as_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "threshold": round(self.threshold, 4),
            "best_threshold": round(self.best_threshold, 4),
            "best_accuracy": round(self.best_accuracy, 4),
            "same_mean": round(self.same_mean, 4),
            "diff_mean": round(self.diff_mean, 4),
            "margin": round(self.margin, 4),
            "pairs": self.pairs,
            "same_pairs": self.same_pairs,
            "valid": self.valid,
        }


@dataclass
class TrainResult:
    projection_path: str | None
    people: int
    train_samples: int
    val_samples: int
    before: EvalMetrics = field(default_factory=EvalMetrics)
    after: EvalMetrics = field(default_factory=EvalMetrics)
    improved: bool = False
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "people": self.people,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "improved": self.improved,
            "message": self.message,
        }


def _crop_for(sample: FaceSample) -> np.ndarray | None:
    """Prefer the saved crop; fall back to re-cutting from the source."""
    if sample.crop_path and Path(sample.crop_path).exists():
        image = cv2.imread(sample.crop_path)
        if image is not None:
            return image
    image = cv2.imread(sample.image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = sample.bbox
    crop = image[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
    return crop if crop.size else None


def embed_samples(
    samples: list[FaceSample], embedder, raw: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Embed every sample; returns (N, D) matrix and person labels.

    `raw=True` bypasses any existing projection so training always starts
    from the base SFace space — otherwise repeated runs would stack
    projections on top of each other.
    """
    vectors: list[np.ndarray] = []
    people: list[str] = []
    for sample in samples:
        crop = _crop_for(sample)
        if crop is None:
            continue
        faces = embedder.detect(crop)
        if faces:
            best = max(faces, key=lambda f: f[-1])
            feature = _embed_raw(embedder, crop, best) if raw else embedder.embed_face(crop, best)
        else:
            # The crop is already a tight face; embed it directly.
            feature = _embed_whole(embedder, crop, raw=raw)
        if feature is None:
            continue
        vectors.append(feature)
        people.append(sample.person)
    if not vectors:
        return np.zeros((0, embedder.dim), dtype=np.float32), []
    return np.vstack(vectors), people


def _embed_raw(embedder, crop, face) -> np.ndarray | None:
    aligned = embedder._recognizer.alignCrop(crop, face)
    feature = embedder._recognizer.feature(aligned).flatten().astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else None


def _embed_whole(embedder, crop, raw: bool) -> np.ndarray | None:
    resized = cv2.resize(crop, (112, 112))
    feature = embedder._recognizer.feature(resized).flatten().astype(np.float32)
    if not raw and getattr(embedder, "_projection", None) is not None:
        feature = feature @ embedder._projection
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else None


def evaluate_embeddings(
    vectors: np.ndarray, people: list[str], threshold: float = 0.36
) -> EvalMetrics:
    """Verification metrics over all same/different pairs."""
    n = len(people)
    if n < 2:
        return EvalMetrics(threshold=threshold)
    sims = vectors @ vectors.T
    iu = np.triu_indices(n, k=1)
    scores = sims[iu]
    labels = np.array(
        [people[i] == people[j] for i, j in zip(*iu, strict=True)], dtype=bool
    )
    if labels.all() or not labels.any():
        # All-same or all-different: AUC is undefined. Return an invalid
        # result rather than 0.0, which would read as "terrible" and, when
        # compared against another invalid 0.0, as "no worse".
        return EvalMetrics(
            threshold=threshold, pairs=len(scores), same_pairs=int(labels.sum())
        )

    same, diff = scores[labels], scores[~labels]
    # ROC AUC via the Mann-Whitney U identity — no sklearn dependency.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[labels].sum() - len(same) * (len(same) + 1) / 2) / (
        len(same) * len(diff)
    )
    predictions = scores >= threshold
    accuracy = float((predictions == labels).mean())

    # Sweep candidate cutoffs to find the best operating point for THIS
    # embedding space, so a fine-tune can report where its threshold
    # should now sit rather than being judged at the old one.
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    accuracies = [(scores >= c).astype(bool) == labels for c in candidates]
    means = np.array([a.mean() for a in accuracies])
    best_i = int(means.argmax())

    return EvalMetrics(
        auc=float(auc),
        accuracy=accuracy,
        threshold=threshold,
        best_threshold=float(candidates[best_i]),
        best_accuracy=float(means[best_i]),
        same_mean=float(same.mean()),
        diff_mean=float(diff.mean()),
        margin=float(same.mean() - diff.mean()),
        pairs=len(scores),
        same_pairs=int(len(same)),
        valid=True,
    )


def train_face_projection(
    train_samples: list[FaceSample],
    val_samples: list[FaceSample],
    embedder,
    *,
    output_dir: str | Path,
    output_dim: int = 128,
    epochs: int = 60,
    lr: float = 1e-3,
    threshold: float = 0.36,
) -> TrainResult:
    """Learn a projection that pulls same-person embeddings together.

    Loss: a proxy-anchor style objective over per-person centroids —
    maximize similarity to your own centroid, minimize it to others,
    with a temperature-scaled softmax. Cheap, stable on small data, and
    directly optimizes the cosine geometry the resolver uses.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_people = embed_samples(train_samples, embedder, raw=True)
    val_x, val_people = embed_samples(val_samples, embedder, raw=True)

    unique = sorted(set(train_people))
    if len(unique) < 2 or len(train_x) < 4:
        return TrainResult(
            projection_path=None,
            people=len(unique),
            train_samples=len(train_x),
            val_samples=len(val_x),
            message=(
                "need at least 2 people and 4 verified face samples to train; "
                "verify more training data first"
            ),
        )

    # Evaluate on held-out data when it can produce a real score; fall
    # back to train only to have *something* to report, never to justify
    # adopting the projection.
    before = evaluate_embeddings(val_x, val_people, threshold)
    eval_on_train = not before.valid
    if eval_on_train:
        before = evaluate_embeddings(train_x, train_people, threshold)

    index = {person: i for i, person in enumerate(unique)}
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor([index[p] for p in train_people], dtype=torch.long)

    dim_in = train_x.shape[1]
    # Initialize near-identity so training starts from the base space and
    # can only improve on it.
    weight = torch.eye(dim_in, output_dim, dtype=torch.float32)
    weight += torch.randn_like(weight) * 0.01
    weight.requires_grad_(True)
    optimizer = torch.optim.Adam([weight], lr=lr)
    temperature = 0.07

    for epoch in range(epochs):
        optimizer.zero_grad()
        projected = torch.nn.functional.normalize(x @ weight, dim=1)
        # Per-person centroids act as learned proxies.
        centroids = torch.stack(
            [projected[y == i].mean(dim=0) for i in range(len(unique))]
        )
        centroids = torch.nn.functional.normalize(centroids, dim=1)
        logits = projected @ centroids.T / temperature
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            log.info("epoch %d/%d loss %.4f", epoch, epochs, loss.item())

    matrix = weight.detach().numpy().astype(np.float32)

    def project(vectors: np.ndarray) -> np.ndarray:
        out = vectors @ matrix
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    eval_x, eval_people = (
        (train_x, train_people) if eval_on_train else (val_x, val_people)
    )
    after = evaluate_embeddings(project(eval_x), eval_people, threshold)

    # Only ship a projection that demonstrably helps on held-out pairs.
    # Two ways this can fail and must NOT adopt:
    #   - the evaluation was degenerate (no same-person pairs), so there
    #     is no evidence either way;
    #   - only training data was evaluable, where a projection trained on
    #     that same data will always look better.
    improved = (
        before.valid and after.valid and not eval_on_train and after.auc >= before.auc
    )
    path = None
    if improved:
        path = output_dir / "face_projection.npy"
        np.save(path, matrix)
        (output_dir / "face_projection.json").write_text(
            json.dumps(
                {
                    "people": unique,
                    "dim_in": dim_in,
                    "dim_out": output_dim,
                    "before": before.as_dict(),
                    "after": after.as_dict(),
                },
                indent=2,
            )
        )
        log.info("saved projection to %s (AUC %.4f -> %.4f)", path, before.auc, after.auc)
    if improved:
        message = "projection saved and now active"
    elif eval_on_train or not after.valid:
        message = (
            "cannot evaluate: the validation set has no same-person pairs. "
            "Verify more samples per person (at least 4 each) so a held-out "
            "split can be scored. Base embeddings kept."
        )
        log.warning(message)
    else:
        message = "no improvement on held-out data; base embeddings kept"
        log.warning(
            "fine-tune did not improve held-out AUC (%.4f -> %.4f); keeping base "
            "embeddings",
            before.auc,
            after.auc,
        )

    return TrainResult(
        projection_path=str(path) if path else None,
        people=len(unique),
        train_samples=len(train_x),
        val_samples=len(val_x),
        before=before,
        after=after,
        improved=improved,
        message=message,
    )


def person_coverage(samples: list[FaceSample]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.person] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
