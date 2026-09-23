"""Load a few configured detection samples and print structural checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import build_dataset_from_config, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_config.yaml")
    parser.add_argument("--role", choices=("source", "target"), default="source")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()
    try:
        dataset = build_dataset_from_config(load_config(args.config), args.role, args.split)
        for index in range(min(args.max_samples, len(dataset))):
            image, target = dataset[index]
            boxes = target["boxes"]
            labels = target["labels"]
            if boxes.shape[0] != labels.shape[0]:
                raise ValueError(f"Sample {index}: boxes and labels have incompatible lengths")
            print(
                f"sample={index} image_shape={tuple(image.shape)} "
                f"boxes={boxes.shape[0]} labels={labels.tolist()}"
            )
        print(f"Dataset smoke test passed: {len(dataset)} samples available")
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"Dataset smoke test could not run: {error}") from error


if __name__ == "__main__":
    main()