"""Dataset and transform utilities for Milestone 2 object detection experiments.

The loaders intentionally require dataset roots and annotation locations from the
configuration. They never download, copy, or modify dataset files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_LABEL_MAP = {
    "person": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "train": 6,
    "motorcycle": 7,
    "bicycle": 8,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_boxes_and_labels(
    boxes: torch.Tensor, labels: torch.Tensor
) -> None:
    """Validate the minimum torchvision detection target contract."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "labels must have shape [N] with the same N as boxes; "
            f"got boxes={tuple(boxes.shape)}, labels={tuple(labels.shape)}"
        )
    if not torch.is_floating_point(boxes):
        raise TypeError("boxes must be a floating-point tensor")
    if labels.dtype != torch.int64:
        raise TypeError("labels must be a torch.int64 tensor")
    if torch.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("box coordinates must satisfy xmax >= xmin and ymax >= ymin")


class Compose:
    """Compose transforms that accept and return ``(image, target)``."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    """Convert a PIL image to a float tensor in CHW format."""

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        image_tensor = image_tensor.contiguous().float().div(255)
        return image_tensor, target


class RandomHorizontalFlip:
    """A detection-safe horizontal flip with configurable probability."""

    def __init__(self, probability: float = 0.5, seed: int | None = None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self._random = random.Random(seed)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        if self._random.random() >= self.probability:
            return image, target
        width, _ = image.size
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes = target["boxes"].clone()
        if boxes.numel():
            old_xmin = boxes[:, 0].clone()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - old_xmin
        target = {**target, "boxes": boxes}
        return flipped, target


def detection_collate_fn(batch):
    """Keep variable-length detection targets as a list."""
    return tuple(zip(*batch))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else base / path


def _image_files(root: Path, split: str | None = None) -> list[Path]:
    search_root = root / split if split and (root / split).is_dir() else root
    return sorted(
        path for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _box_from_points(points: Iterable[Sequence[float]]) -> list[float] | None:
    coordinates = list(points)
    if not coordinates:
        return None
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    return [min(xs), min(ys), max(xs), max(ys)]


def _cityscapes_objects(annotation: Mapping[str, Any]) -> list[tuple[str, list[float]]]:
    objects = []
    for item in annotation.get("objects", []):
        label = item.get("label")
        box = item.get("bbox")
        if box is None:
            box = _box_from_points(item.get("polygon", []))
        if label and box and len(box) == 4:
            objects.append((str(label), [float(value) for value in box]))
    return objects


def _target_from_objects(
    objects: Iterable[tuple[str, Sequence[float]]],
    label_map: Mapping[str, int],
    image_id: int,
) -> dict[str, Any]:
    boxes = []
    labels = []
    for label_name, box in objects:
        label_id = label_name if isinstance(label_name, int) else label_map.get(label_name)
        if label_id is not None and len(box) == 4:
            boxes.append([float(value) for value in box])
            labels.append(label_id)
    box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    validate_boxes_and_labels(box_tensor, label_tensor)
    return {
        "boxes": box_tensor,
        "labels": label_tensor,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
    }


class CityscapesDataset(Dataset):
    """Load Cityscapes or Foggy Cityscapes images and polygon JSON labels.

    Annotation matching supports either the same relative path with a ``.json``
    suffix or a recursive stem match. The latter accommodates common derived
    annotation layouts without assuming one machine-specific directory tree.
    """

    def __init__(
        self,
        image_root: str | Path,
        annotation_root: str | Path,
        split: str | None = None,
        label_map: Mapping[str, int] | None = None,
        transforms: Callable | None = None,
    ):
        self.image_root = Path(image_root).expanduser()
        self.annotation_root = Path(annotation_root).expanduser()
        self.split = split
        self.label_map = dict(label_map or DEFAULT_LABEL_MAP)
        self.transforms = transforms or Compose([ToTensor()])
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Cityscapes image root does not exist: {self.image_root}")
        if not self.annotation_root.is_dir():
            raise FileNotFoundError(
                f"Cityscapes annotation root does not exist: {self.annotation_root}"
            )
        self.images = _image_files(self.image_root, split)
        if not self.images:
            raise FileNotFoundError(f"No images found below {self.image_root}")
        self._annotations = sorted(self.annotation_root.rglob("*.json"))

    def __len__(self) -> int:
        return len(self.images)

    def _annotation_for(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_root)
        direct_candidates = [self.annotation_root / relative.with_suffix(".json")]
        direct_candidates.append(self.annotation_root / image_path.name.replace(image_path.suffix, ".json"))
        for candidate in dict.fromkeys(direct_candidates):
            if candidate.is_file():
                return candidate

        # The release-specific relative mapping is intentionally not guessed.
        # A recursive fallback is safe only when exactly one exact-stem JSON
        # exists; otherwise the dataset layout must be configured/verified.
        fallback_matches = [
            path for path in self._annotations if path.stem == image_path.stem
        ]
        if len(fallback_matches) == 1:
            return fallback_matches[0]
        if len(fallback_matches) > 1:
            formatted = ", ".join(str(path) for path in fallback_matches)
            raise RuntimeError(
                f"Ambiguous annotations for image {image_path}: found "
                f"{len(fallback_matches)} exact-stem matches ({formatted}). "
                "Verify annotation_root and the Cityscapes/Foggy Cityscapes layout."
            )
        raise FileNotFoundError(
            f"No deterministic annotation JSON found for image {image_path}. "
            "Verify annotation_root and the Cityscapes/Foggy Cityscapes layout/configuration."
        )

    def __getitem__(self, index: int):
        image_path = self.images[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        annotation = _read_json(self._annotation_for(image_path))
        target = _target_from_objects(
            _cityscapes_objects(annotation), self.label_map, index
        )
        return self.transforms(image, target)


def _bdd_objects(annotation: Any) -> list[tuple[str, list[float]]]:
    """Extract common BDD label structures without assuming one label file shape."""
    if isinstance(annotation, list):
        objects = annotation
    elif isinstance(annotation, Mapping):
        objects = annotation.get("labels", annotation.get("objects", []))
        if not objects and "frames" in annotation:
            objects = [obj for frame in annotation["frames"] for obj in frame.get("objects", [])]
    else:
        objects = []
    parsed = []
    for item in objects:
        name = item.get("category", item.get("name", item.get("label")))
        box = item.get("box2d", item.get("bbox"))
        if isinstance(box, Mapping):
            box = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
        if name and box and len(box) == 4 and all(value is not None for value in box):
            parsed.append((str(name), [float(value) for value in box]))
    return parsed


class BDD100KSubsetDataset(Dataset):
    """Load images selected by ``create_bdd100k_subset.py`` manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        images_root: str | Path | None = None,
        labels_path: str | Path | None = None,
        label_map: Mapping[str, int] | None = None,
        split: str | None = None,
        transforms: Callable | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        self.base_dir = self.manifest_path.parent
        self.images_root = Path(images_root).expanduser() if images_root else None
        self.labels_path = Path(labels_path).expanduser() if labels_path else None
        self.label_map = dict(label_map or DEFAULT_LABEL_MAP)
        self.transforms = transforms or Compose([ToTensor()])
        self.records = _read_manifest(self.manifest_path)
        if split:
            self.records = [record for record in self.records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"No records found in manifest for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, record: Mapping[str, Any]) -> Path:
        path = _resolve_path(record["image_path"], self.base_dir)
        if not path.is_file() and self.images_root:
            path = _resolve_path(record["image_path"], self.images_root)
        return path

    def _annotation(self, record: Mapping[str, Any]) -> Any:
        if "objects" in record or "boxes" in record:
            return record
        annotation_path = record.get("annotation_path")
        if annotation_path:
            return _read_json(_resolve_path(annotation_path, self.base_dir))
        if self.labels_path:
            labels = _read_json(self.labels_path)
            if isinstance(labels, list):
                image_name = Path(record["image_path"]).name
                matches = [item for item in labels if item.get("name") == image_name]
                if matches:
                    return matches[0]
        raise FileNotFoundError(
            f"No BDD annotation for manifest record {record.get('image_path')}"
        )

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self._image_path(record)
        if not image_path.is_file():
            raise FileNotFoundError(f"BDD image does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        annotation = self._annotation(record)
        if "boxes" in annotation:
            objects = zip(annotation.get("labels", []), annotation["boxes"])
        else:
            objects = _bdd_objects(annotation)
        target = _target_from_objects(objects, self.label_map, index)
        return self.transforms(image, target)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BDD subset manifest does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = _read_json(path)
        return data if isinstance(data, list) else data.get("records", [])
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid manifest JSON on line {line_number}") from error
    return records


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def build_dataset_from_config(
    config: Mapping[str, Any],
    role: str = "source",
    split: str | None = None,
    transforms: Callable | None = None,
) -> Dataset:
    """Build the configured source or target dataset without hard-coded paths."""
    if role not in {"source", "target"}:
        raise ValueError("role must be 'source' or 'target'")
    domain_name = config.get("source_domain") if role == "source" else config.get("target_domain")
    domains = config.get("domains", {})
    domain = domains.get(domain_name, {})
    dataset_type = str(domain.get("type", domain_name)).lower()
    dataset_root = domain.get("root", config.get("dataset_root"))
    selected_split = split or config.get("split", {}).get("train", "train")
    if dataset_type in {"cityscapes", "foggy_cityscapes"}:
        if not dataset_root or not domain.get("annotation_root"):
            raise ValueError(
                f"Configure verified root and annotation_root for domain {domain_name!r}"
            )
        return CityscapesDataset(
            dataset_root,
            domain.get("annotation_root"),
            split=selected_split,
            transforms=transforms,
        )
    if dataset_type in {"bdd100k", "bdd100k_subset"}:
        manifest = config.get("bdd100k_subset_manifest", domain.get("manifest"))
        if not manifest:
            raise ValueError("Configure bdd100k_subset_manifest before loading BDD100K")
        return BDD100KSubsetDataset(
            manifest,
            images_root=domain.get("images_root"),
            labels_path=domain.get("labels_path"),
            split=selected_split,
            transforms=transforms,
        )
    raise ValueError(f"Unsupported dataset type/domain: {dataset_type!r}")


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )