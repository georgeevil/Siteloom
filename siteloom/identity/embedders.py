"""Embedding algorithms behind the identifier framework.

Two families, reflecting how unevenly identification research has
matured (the reason identifiers are per-class configurable):

- FaceEmbedder: a dedicated face pipeline — detection (YuNet), landmark
  alignment, then a purpose-trained face recognition embedding (SFace).
  Face ID has decades of work behind it; using a generic appearance
  embedding here would throw that away.
- GenericEmbedder: an ImageNet-backbone appearance embedding (ResNet-18
  global features). Much weaker as an identifier, but it works for ANY
  crop — person, vehicle, or a class added dynamically at runtime —
  which is exactly its job (PRD §6.4 vehicle re-ID for obscured plates).

Both expose `embed(bgr) -> np.ndarray | None` (L2-normalized) and `dim`,
so the resolver treats them uniformly; only thresholds differ.

The face weights are downloaded rather than vendored, so every file that
reaches `cv2.FaceRecognizerSF` is pinned by SHA-256 and verified on every
load — cached or fresh (CLD-50). Weights that decide who a face belongs
to are worth the hash; a size heuristic only catches a truncated file.

Embedders are backend-blind and stateless after warm-up, matching the
processing-module rules (PRD §7).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path.home() / ".cache" / "siteloom" / "models"
# Pre-seeded/offline installs point this at a directory they control.
MODELS_DIR_ENV = "SITELOOM_MODELS_DIR"

# opencv_zoo stores models in git-lfs; media.githubusercontent resolves
# the actual blobs (raw.githubusercontent would return LFS pointers).
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class ModelSpec:
    """A weights file we are willing to load, named by its content.

    The cache is a directory in the operator's home that any local
    process can write, and these weights decide who the system thinks a
    face belongs to — so the identity of the artifact is the digest, not
    the URL it once came from and not its size. `size` is carried too
    because it is the cheap pre-check that rules out the common failure
    (a truncated download, or git-lfs's 130-byte pointer file) without
    reading 39 MB.
    """

    url: str
    sha256: str
    size: int

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


# Digests verified against opencv_zoo's git-lfs pointers (the `oid
# sha256:` line raw.githubusercontent serves for each path), which is an
# independent source from the blob download itself. See docs/operations.md
# before changing either of these.
YUNET = ModelSpec(
    url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232_589,
)
SFACE = ModelSpec(
    url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38_696_353,
)
FACE_MODELS = (YUNET, SFACE)


class ModelIntegrityError(Exception):
    """A weights file is not the artifact we pinned.

    Raised rather than downgraded to a warning: loading unverified face
    weights is exactly the thing this check exists to prevent.
    """


def models_dir() -> Path:
    """Where face weights live. `SITELOOM_MODELS_DIR` overrides the cache."""
    override = os.environ.get(MODELS_DIR_ENV)
    return Path(override).expanduser() if override else DEFAULT_MODELS_DIR


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def mismatch(path: Path, spec: ModelSpec) -> str | None:
    """Why `path` is not `spec`, or None if it is. Never raises for a
    file that is merely wrong — only for one that cannot be read."""
    size = path.stat().st_size
    if size != spec.size:
        return f"size {size} B, expected {spec.size} B"
    digest = sha256_file(path)
    if digest != spec.sha256:
        return f"sha256 {digest}, expected {spec.sha256}"
    return None


def _recovery_advice(spec: ModelSpec, path: Path) -> str:
    return (
        "The file has been deleted, so the next run re-downloads it. If a clean "
        "re-download fails the same way, the download is not corrupt — upstream "
        "republished the artifact. Do not edit the pinned digest to make this "
        f"pass: check {spec.filename} against opencv_zoo's git-lfs pointer "
        f"(the 'oid sha256:' line at "
        f"https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/...) "
        f"and repin deliberately. To install without network access, pre-seed "
        f"{path.parent} with a verified copy or point {MODELS_DIR_ENV} at one."
    )


def _fetch(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream `url` to `dest`. Split out so tests can make the network fail."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with dest.open("wb") as out:
            shutil.copyfileobj(response, out)


def ensure_model(spec: ModelSpec, directory: Path | str | None = None) -> Path:
    """Return a local path to `spec`, verified by digest.

    A cached file is verified on every call, not just on the run that
    downloaded it: verifying only at download time means the check is
    skipped on every run after the first, which is every run that
    matters. A file that fails is removed before anything else happens —
    leaving it behind would mean the next run loads it without even
    trying to re-download.

    A cached file that verifies is returned without touching the
    network, so an air-gapped install with a pre-seeded directory works.
    """
    directory = Path(directory) if directory is not None else models_dir()
    dest = directory / spec.filename

    if dest.exists():
        problem = mismatch(dest, spec)
        if problem is None:
            return dest
        dest.unlink()
        log.warning(
            "cached %s failed verification (%s) and was deleted; re-downloading",
            dest,
            problem,
        )

    directory.mkdir(parents=True, exist_ok=True)
    # Download beside the target and rename only once verified, so the
    # cache never holds a half-written or unverified file at any point.
    # The pid keeps two processes warming up at once (serve and frigate
    # both build a FaceEmbedder) out of each other's staging file; the
    # rename onto `dest` is atomic, so whichever finishes last wins.
    staging = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    staging.unlink(missing_ok=True)
    log.info("downloading %s", spec.url)
    try:
        _fetch(spec.url, staging)
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise IOError(
            f"could not download {spec.url}: {type(exc).__name__}: {exc}. "
            f"To install without network access, pre-seed {directory} with a "
            f"verified copy of {spec.filename} or point {MODELS_DIR_ENV} at a "
            f"directory containing one."
        ) from exc

    problem = mismatch(staging, spec)
    if problem is not None:
        staging.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"{spec.filename} downloaded from {spec.url} failed its integrity "
            f"check: {problem}. {_recovery_advice(spec, dest)}"
        )
    staging.replace(dest)
    return dest


@dataclass(frozen=True)
class ModelStatus:
    """What a diagnostic can say about one cached file without changing it."""

    spec: ModelSpec
    path: Path
    state: str  # "ok" | "missing" | "corrupt" | "unreadable"
    detail: str = ""


def check_cached_model(spec: ModelSpec, directory: Path | str | None = None) -> ModelStatus:
    """Verify one cached file in place. Read-only: a diagnostic reports,
    it does not delete — `ensure_model` is what cleans up."""
    directory = Path(directory) if directory is not None else models_dir()
    path = directory / spec.filename
    if not path.exists():
        return ModelStatus(spec, path, "missing")
    try:
        problem = mismatch(path, spec)
    except OSError as exc:
        return ModelStatus(spec, path, "unreadable", f"{type(exc).__name__}: {exc}")
    if problem is None:
        return ModelStatus(spec, path, "ok")
    return ModelStatus(spec, path, "corrupt", problem)


class FaceEmbedder:
    """YuNet face detection + SFace 128-d embedding, both in OpenCV core.

    Runs on CPU (ONNX); fast enough for crop-sized inputs. Swappable for
    InsightFace/ArcFace later behind the same `embed()` shape.

    Detection and embedding are exposed separately as well as combined,
    because the Takeout importer and training exporter need face *boxes*
    on full photos, not just one embedding per person crop.

    If a fine-tuned projection head exists (see siteloom/training/face.py)
    it is applied to the raw SFace feature, so training improvements take
    effect everywhere embeddings are computed with no call-site changes.
    """

    dim = 128
    # A face smaller than this in either dimension is not worth embedding.
    MIN_FACE_PX = 32

    def __init__(
        self,
        projection_path: str | Path | None = None,
        models_directory: Path | str | None = None,
    ) -> None:
        det_path = ensure_model(YUNET, models_directory)
        rec_path = ensure_model(SFACE, models_directory)
        self._detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), score_threshold=0.7
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._projection = _load_projection(projection_path)
        if self._projection is not None:
            self.dim = self._projection.shape[1]

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """All faces in an image. Each row is YuNet's 15-value format:
        [x, y, w, h, 5x landmark xy..., score]."""
        h, w = bgr.shape[:2]
        if h < self.MIN_FACE_PX or w < self.MIN_FACE_PX:
            return []
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return []
        return [face for face in faces]

    def embed_face(self, bgr: np.ndarray, face: np.ndarray) -> np.ndarray | None:
        """Embed one already-detected face using its landmarks to align."""
        aligned = self._recognizer.alignCrop(bgr, face)
        feature = self._recognizer.feature(aligned).flatten().astype(np.float32)
        return self._finish(feature)

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        """Best face in the image, embedded — the ProcessingModule path."""
        faces = self.detect(bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f[-1])
        return self.embed_face(bgr, best)

    def _finish(self, feature: np.ndarray) -> np.ndarray | None:
        if self._projection is not None:
            feature = feature @ self._projection
        norm = np.linalg.norm(feature)
        return (feature / norm).astype(np.float32) if norm > 0 else None


def _load_projection(path: str | Path | None) -> np.ndarray | None:
    """Load a fine-tuned projection matrix if one has been trained."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    matrix = np.load(path).astype(np.float32)
    log.info("using fine-tuned face projection %s %s", path, matrix.shape)
    return matrix


class GenericEmbedder:
    """ResNet-18 penultimate-layer features (512-d), ImageNet weights.

    One shared instance serves every generic identifier regardless of
    class — the class separation happens in the vector store collections,
    not the embedder.
    """

    dim = 512

    def __init__(self, device: str = "mps") -> None:
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self._device = _resolve_device(torch, device)
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self._net = net.eval().to(self._device)
        self._prep = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        if bgr.shape[0] < 16 or bgr.shape[1] < 16:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._torch.no_grad():
            batch = self._prep(rgb).unsqueeze(0).to(self._device)
            feature = self._net(batch).squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else None


def _resolve_device(torch, requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_embedder(algo: str, device: str = "mps", projection_path=None):
    if algo == "face":
        return FaceEmbedder(projection_path=projection_path)
    if algo == "generic":
        return GenericEmbedder(device=device)
    raise ValueError(f"unknown embedding algo {algo!r}")
