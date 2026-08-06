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

Embedders are backend-blind and stateless after warm-up, matching the
processing-module rules (PRD §7).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "siteloom" / "models"

# opencv_zoo stores models in git-lfs; media.githubusercontent resolves
# the actual blobs (raw.githubusercontent would return LFS pointers).
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
YUNET_URL = f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def _download(url: str, min_bytes: int) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / url.rsplit("/", 1)[-1]
    if not dest.exists() or dest.stat().st_size < min_bytes:
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < min_bytes:
            dest.unlink(missing_ok=True)
            raise IOError(f"download of {url} looks truncated (LFS pointer?)")
    return dest


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

    def __init__(self, projection_path: str | Path | None = None) -> None:
        det_path = _download(YUNET_URL, min_bytes=100_000)
        rec_path = _download(SFACE_URL, min_bytes=10_000_000)
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
