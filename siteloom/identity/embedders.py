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
    """

    dim = 128

    def __init__(self) -> None:
        det_path = _download(YUNET_URL, min_bytes=100_000)
        rec_path = _download(SFACE_URL, min_bytes=10_000_000)
        self._detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), score_threshold=0.7
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        if h < 40 or w < 40:
            return None  # too small to carry a usable face
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # Best-scoring face in the crop (a person crop should hold one).
        face = faces[np.argmax(faces[:, -1])]
        aligned = self._recognizer.alignCrop(bgr, face)
        feature = self._recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else None


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


def build_embedder(algo: str, device: str = "mps"):
    if algo == "face":
        return FaceEmbedder()
    if algo == "generic":
        return GenericEmbedder(device=device)
    raise ValueError(f"unknown embedding algo {algo!r}")
