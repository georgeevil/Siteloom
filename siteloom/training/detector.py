"""YOLO face-detector training from verified annotations.

What this does and does not buy you, stated plainly: training a YOLO
face detector improves face *detection* — recall on small, angled, or
motion-blurred faces in your own footage, and speed on GPU — but it does
nothing for *identification*. Who a face belongs to remains the job of
the embedding + vector-store path.

It is still worth having: YuNet is tuned for reasonably frontal, clean
faces, and a detector trained on frames from your actual cameras and
archive will find faces YuNet misses. Those extra faces then flow into
the identification path that already exists.

Training data is only ever verified face annotations (see dataset.py),
exported in YOLO format with a person-disjoint... note: *image*-disjoint
train/val split — the same photo must never appear in both halves.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.store import Annotation, LibraryItem

log = logging.getLogger(__name__)


@dataclass
class ExportResult:
    dataset_yaml: str
    train_images: int
    val_images: int
    boxes: int
    message: str = ""


def export_yolo_dataset(
    session: Session,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    class_names: tuple[str, ...] = ("face",),
) -> ExportResult:
    """Write verified boxes as a YOLO dataset (images + label txt files).

    Images are symlinked rather than copied — a personal photo archive is
    large, and the training run only needs to read them.
    """
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = output_dir / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    wanted = set(class_names)
    class_index = {name: i for i, name in enumerate(class_names)}

    rows = session.scalars(
        select(Annotation).filter(
            Annotation.class_name.in_(wanted),
            Annotation.verified.is_(True),
            Annotation.rejected.is_(False),
        )
    ).all()

    # Group by item so an image and all of its boxes stay together, and
    # so the split is image-disjoint.
    by_item: dict[int, list[Annotation]] = defaultdict(list)
    for annotation in rows:
        by_item[annotation.item_id].append(annotation)

    item_ids = sorted(by_item)
    n_val = int(len(item_ids) * val_fraction)
    val_ids = set(item_ids[:n_val])

    train_images = val_images = boxes = 0
    for item_id in item_ids:
        item = session.get(LibraryItem, item_id)
        if item is None or item.kind != "image":
            continue
        source = Path(item.path)
        if not source.exists():
            continue
        split = "val" if item_id in val_ids else "train"
        dest = output_dir / "images" / split / f"{item_id}{source.suffix}"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(source)
        except OSError:
            shutil.copy2(source, dest)

        lines = []
        for annotation in by_item[item_id]:
            x1, y1, x2, y2 = json.loads(annotation.bbox)
            # YOLO wants normalized center-x, center-y, width, height.
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            lines.append(
                f"{class_index[annotation.class_name]} "
                f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )
            boxes += 1
        (output_dir / "labels" / split / f"{item_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        if split == "val":
            val_images += 1
        else:
            train_images += 1

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {i}: {name}" for name, i in class_index.items()],
                "",
            ]
        )
    )
    message = ""
    if train_images == 0:
        message = (
            "no verified face annotations on image items — verify training data "
            "in the review UI first"
        )
    elif val_images == 0:
        message = "too few images for a validation split; all went to train"
    return ExportResult(
        dataset_yaml=str(yaml_path),
        train_images=train_images,
        val_images=val_images,
        boxes=boxes,
        message=message,
    )


def train_face_detector(
    dataset_yaml: str | Path,
    *,
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "mps",
    project: str | Path = "training/detector",
) -> dict:
    """Fine-tune YOLO on the exported face dataset."""
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name="face",
        exist_ok=True,
        verbose=False,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project)
    weights = save_dir / "weights" / "best.pt"
    metrics = {}
    box = getattr(getattr(results, "box", None), "map50", None)
    if box is not None:
        metrics["mAP50"] = float(box)
        metrics["mAP50-95"] = float(results.box.map)
    return {"weights": str(weights), "metrics": metrics, "save_dir": str(save_dir)}
