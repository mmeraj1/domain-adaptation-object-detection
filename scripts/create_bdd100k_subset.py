"""Create a manifest-only, deterministic BDD100K development subset.

This script reads metadata supplied by the user and never copies or modifies
the original BDD100K images or annotations. Sampling targets approximately
equal representation of rain, snow, and night while treating conditions as
overlapping metadata dimensions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CONDITIONS = ("rain", "snow", "night")
IMAGE_KEYS = ("image_path", "image", "name", "file_name", "filename")
SPLIT_KEYS = ("split", "partition", "set")


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
    searchable = weather + time_of_day
    return [condition for condition in CONDITIONS if any(condition in value for value in searchable)]


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
        normalized.append(
            {
                "image_path": image_path,
                "conditions": extract_conditions(record),
                "source_metadata": dict(record),
                **({"split": _record_split(record)} if _record_split(record) else {}),
            }
        )
    return normalized


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
    remaining = list(records)
    rng.shuffle(remaining)
    quotas = {condition: target_count / len(CONDITIONS) for condition in CONDITIONS}
    counts = {condition: 0 for condition in CONDITIONS}
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < target_count:
        scores = [
            _sampling_score(record, counts, quotas) for record in remaining
        ]
        best_score = min(scores)
        best_indices = [index for index, score in enumerate(scores) if score == best_score]
        chosen_index = rng.choice(best_indices)
        chosen = remaining.pop(chosen_index)
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
    return {
        "total_metadata_records": total_metadata_records,
        "adverse_condition_candidate_records": candidate_records,
        "selected_images": len(records),
        "by_condition": {condition: condition_counts[condition] for condition in CONDITIONS},
        "overlap_counts": overlap_counts,
        "counts_by_original_split": dict(
            Counter(record["split"] for record in original_split_records)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="User-provided BDD metadata JSON/JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest path")
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
    raw_records = read_records(args.metadata)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    stats = summarize(selected, len(raw_records), len(candidates))
    stats["seed"] = args.seed
    stats["target_count"] = args.target_count
    stats["fraction"] = args.fraction
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()