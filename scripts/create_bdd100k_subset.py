"""Create a deterministic BDD100K development subset and manifest.

This script reads metadata supplied by the user, copies only selected images
into a separate subset root, and never modifies the original BDD100K files.
Sampling targets approximately equal representation of rain, snow, and night
while treating conditions as overlapping metadata dimensions.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CONDITIONS = ("rain", "snow", "night")
IMAGE_KEYS = ("image_path", "image", "name", "file_name", "filename")
SPLIT_KEYS = ("split", "partition", "set")
CONDITION_NORMALIZATION = {
    "rainy": "rain",
    "rain": "rain",
    "snowy": "snow",
    "snow": "snow",
    "night": "night",
}
CLASS_NORMALIZATION = {"bike": "bicycle", "motor": "motorcycle"}
CANONICAL_CLASSES = {
    "person": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "train": 6,
    "motorcycle": 7,
    "bicycle": 8,
}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array/object or JSONL metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL at line {line_number}") from error
        return records
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("records", "images", "metadata", "data"):
            if isinstance(data.get(key), list):
                return [record for record in data[key] if isinstance(record, dict)]
    raise ValueError("Metadata must be a JSON list, JSONL file, or object containing records/images")


def _find_value(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    wanted = {key.lower() for key in keys}
    stack = [record]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if str(key).lower() in wanted and value not in (None, ""):
                    return value
                if isinstance(value, (Mapping, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_strings(child)]
    return []


def extract_image_path(record: Mapping[str, Any]) -> str | None:
    value = _find_value(record, IMAGE_KEYS)
    return str(value) if value is not None else None


def extract_conditions(record: Mapping[str, Any]) -> list[str]:
    weather = _flatten_strings(_find_value(record, ("weather", "weather_condition")))
    time_of_day = _flatten_strings(
        _find_value(record, ("timeofday", "time_of_day", "time", "scene_time"))
    )
    normalized = {
        CONDITION_NORMALIZATION[value]
        for value in weather + time_of_day
        if value in CONDITION_NORMALIZATION
    }
    return [condition for condition in CONDITIONS if condition in normalized]


def _record_split(record: Mapping[str, Any]) -> str | None:
    value = _find_value(record, SPLIT_KEYS)
    return str(value).lower() if value is not None else None


def normalize_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize image identity and condition metadata while preserving source fields."""
    normalized = []
    seen = set()
    for record in records:
        image_path = extract_image_path(record)
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        normalized_record = {
            "image_path": image_path,
            "conditions": extract_conditions(record),
            "source_metadata": dict(record),
        }
        split = _record_split(record)
        if split:
            normalized_record["split"] = split
        normalized_record.update(_canonical_detection(record))
        normalized.append(normalized_record)
    return normalized


def _canonical_detection(record: Mapping[str, Any]) -> dict[str, Any]:
    boxes = []
    labels = []
    normalization_counts = {"bike_to_bicycle": 0, "motor_to_motorcycle": 0}
    invalid_box_count = 0
    for obj in record.get("labels", []) or []:
        if not isinstance(obj, Mapping):
            continue
        box = obj.get("box2d")
        category = obj.get("category")
        if not isinstance(box, Mapping) or category not in CANONICAL_CLASSES:
            if category in CLASS_NORMALIZATION:
                category = CLASS_NORMALIZATION[category]
                if category not in CANONICAL_CLASSES:
                    continue
            else:
                continue
        values = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
        if len(values) != 4 or any(value is None for value in values):
            invalid_box_count += 1
            continue
        try:
            numeric_box = [float(value) for value in values]
        except (TypeError, ValueError):
            invalid_box_count += 1
            continue
        if numeric_box[2] < numeric_box[0] or numeric_box[3] < numeric_box[1]:
            invalid_box_count += 1
            continue
        original_category = obj.get("category")
        if original_category == "bike":
            normalization_counts["bike_to_bicycle"] += 1
        elif original_category == "motor":
            normalization_counts["motor_to_motorcycle"] += 1
        boxes.append(numeric_box)
        labels.append(category)
    return {
        "boxes": boxes,
        "labels": labels,
        "class_normalization_counts": normalization_counts,
        "invalid_box_count": invalid_box_count,
    }


def _sampling_score(
    record: Mapping[str, Any],
    counts: Mapping[str, int],
    quotas: Mapping[str, float],
) -> float:
    """Score a candidate by the post-selection distance from condition quotas."""
    score = 0.0
    conditions = set(record["conditions"])
    for condition in CONDITIONS:
        next_count = counts[condition] + int(condition in conditions)
        score += (next_count - quotas[condition]) ** 2
    return score


def _sample_condition_balanced(
    records: list[dict[str, Any]], target_count: int, seed: int
) -> list[dict[str, Any]]:
    """Select unique records while minimizing deviation from equal condition quotas.

    Every selected image is removed from the candidate pool after selection. A
    multi-condition image contributes to every condition it contains, so overlap
    is preserved rather than being forced into an exclusive class.
    """
    if target_count >= len(records):
        return list(records)
    rng = random.Random(seed)
    remaining = set(range(len(records)))
    condition_indices = {
        condition: [
            index for index, record in enumerate(records) if condition in record["conditions"]
        ]
        for condition in CONDITIONS
    }
    for indices in condition_indices.values():
        rng.shuffle(indices)
    quotas = {condition: target_count / len(CONDITIONS) for condition in CONDITIONS}
    counts = {condition: 0 for condition in CONDITIONS}
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < target_count:
        deficits = {
            condition: quotas[condition] - counts[condition] for condition in CONDITIONS
        }
        condition = max(CONDITIONS, key=lambda item: deficits[item])
        available = [index for index in condition_indices[condition] if index in remaining]
        if not available:
            available = list(remaining)
            rng.shuffle(available)
        best_deficit = max(
            sum(max(deficits[item], 0.0) for item in records[index]["conditions"])
            for index in available
        )
        best_indices = [
            index
            for index in available
            if sum(max(deficits[item], 0.0) for item in records[index]["conditions"])
            == best_deficit
        ]
        chosen_index = rng.choice(best_indices)
        remaining.remove(chosen_index)
        chosen = records[chosen_index]
        selected.append(chosen)
        for condition in chosen["conditions"]:
            counts[condition] += 1
    return selected


def select_subset(
    records: list[dict[str, Any]],
    target_count: int = 10000,
    seed: int = 42,
    fraction: float | None = None,
) -> list[dict[str, Any]]:
    """Select unique adverse-condition records with approximately equal coverage.

    ``target_count`` is the primary interface. ``fraction`` is an optional
    alternative based on the number of adverse-condition candidates. Existing
    source split fields are retained; no split is generated when absent.
    """
    if target_count <= 0:
        raise ValueError("target_count must be a positive integer")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    candidates = [record for record in records if record.get("conditions")]
    requested_count = round(len(candidates) * fraction) if fraction is not None else target_count
    requested_count = max(1, requested_count)
    return _sample_condition_balanced(candidates, min(requested_count, len(candidates)), seed)


def summarize(
    records: Iterable[Mapping[str, Any]],
    total_metadata_records: int,
    candidate_records: int,
) -> dict[str, Any]:
    records = list(records)
    condition_counts = Counter(
        condition for record in records for condition in record.get("conditions", [])
    )
    overlap_counts = {
        "rain_and_snow": sum(
            {"rain", "snow"}.issubset(record.get("conditions", [])) for record in records
        ),
        "rain_and_night": sum(
            {"rain", "night"}.issubset(record.get("conditions", [])) for record in records
        ),
        "snow_and_night": sum(
            {"snow", "night"}.issubset(record.get("conditions", [])) for record in records
        ),
        "rain_and_snow_and_night": sum(
            set(CONDITIONS).issubset(record.get("conditions", [])) for record in records
        ),
    }
    original_split_records = [record for record in records if "split" in record]
    class_normalization_counts = Counter()
    invalid_box_count = 0
    for record in records:
        class_normalization_counts.update(record.get("class_normalization_counts", {}))
        invalid_box_count += int(record.get("invalid_box_count", 0))
    return {
        "total_metadata_records": total_metadata_records,
        "adverse_condition_candidate_records": candidate_records,
        "selected_images": len(records),
        "by_condition": {condition: condition_counts[condition] for condition in CONDITIONS},
        "overlap_counts": overlap_counts,
        "counts_by_original_split": dict(
            Counter(record["split"] for record in original_split_records)
        ),
        "class_normalization_counts": dict(class_normalization_counts),
        "invalid_bounding_box_count": invalid_box_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        nargs="+",
        required=True,
        help="One or more BDD annotation JSON/JSONL files",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest path")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument(
        "--target-count",
        type=int,
        default=10000,
        help="Approximate number of unique adverse-condition images to select (default: 10000)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional alternative: fraction of adverse-condition candidates to select",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_records = []
    for metadata_path in args.metadata:
        split_name = "train" if "train" in metadata_path.name.lower() else "val"
        for record in read_records(metadata_path):
            if not _record_split(record):
                record = {**record, "split": split_name}
            raw_records.append(record)
    records = normalize_records(raw_records)
    if not records:
        raise SystemExit("No unique image records with recognizable image paths were found")
    candidates = [record for record in records if record.get("conditions")]
    selected = select_subset(
        records,
        target_count=args.target_count,
        seed=args.seed,
        fraction=args.fraction,
    )
    source_indexes = {
        split: {path.name: path for path in (args.images_root / split).rglob("*.jpg")}
        for split in {str(record.get("split")) for record in selected}
    }
    selected_paths = set()
    missing_image_count = 0
    duplicate_count = 0
    for record in selected:
        source_path = source_indexes.get(str(record["split"]), {}).get(
            Path(record["image_path"]).name
        )
        destination_path = args.subset_root / str(record["split"]) / Path(record["image_path"]).name
        if record["image_path"] in selected_paths:
            duplicate_count += 1
            continue
        selected_paths.add(record["image_path"])
        if source_path is None:
            missing_image_count += 1
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    stats = summarize(selected, len(raw_records), len(candidates))
    stats["seed"] = args.seed
    stats["target_count"] = args.target_count
    stats["fraction"] = args.fraction
    stats["duplicate_count"] = duplicate_count
    stats["missing_image_count"] = missing_image_count
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()