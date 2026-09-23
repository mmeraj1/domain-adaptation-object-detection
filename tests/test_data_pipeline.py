import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from scripts.create_bdd100k_subset import normalize_records, select_subset, summarize
from src.data.data_loader import CityscapesDataset, load_config, validate_boxes_and_labels


class TestDataPipeline(unittest.TestCase):
    def test_validate_boxes_and_labels_accepts_detection_target(self):
        validate_boxes_and_labels(
            torch.tensor([[0.0, 1.0, 10.0, 11.0]]),
            torch.tensor([3], dtype=torch.int64),
        )

    def test_validate_boxes_and_labels_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "same N"):
            validate_boxes_and_labels(
                torch.zeros((2, 4)), torch.tensor([1], dtype=torch.int64)
            )

    def test_bdd_selection_is_deterministic_balanced_and_multilabel(self):
        records = normalize_records(
            [{"name": f"rain_{index}.jpg", "weather": "rain"} for index in range(30)]
            + [{"name": f"snow_{index}.jpg", "weather": "snow"} for index in range(30)]
            + [{"name": f"night_{index}.jpg", "timeofday": "night"} for index in range(30)]
            + [{"name": "rain_night.jpg", "weather": "rain", "timeofday": "night"}]
        )
        first = select_subset(records, target_count=30, seed=7)
        second = select_subset(records, target_count=30, seed=7)
        self.assertEqual(
            [item["image_path"] for item in first],
            [item["image_path"] for item in second],
        )
        summary = summarize(first, total_metadata_records=len(records), candidate_records=len(records))
        counts = summary["by_condition"]
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(summary["overlap_counts"]["rain_and_night"], 1)
        self.assertEqual(len({item["image_path"] for item in first}), len(first))
        self.assertTrue(all("split" not in item for item in first))

    def test_bdd_selection_preserves_existing_splits_without_inventing_them(self):
        records = normalize_records(
            [
                {"name": "rain.jpg", "weather": "rain", "split": "official_train"},
                {"name": "snow.jpg", "weather": "snow", "split": "official_val"},
                {"name": "night.jpg", "timeofday": "night", "split": "official_test"},
            ]
        )
        selected = select_subset(records, target_count=3, seed=42)
        self.assertEqual(
            {item["split"] for item in selected},
            {"official_train", "official_val", "official_test"},
        )
        summary = summarize(selected, total_metadata_records=3, candidate_records=3)
        self.assertEqual(
            summary["counts_by_original_split"],
            {"official_train": 1, "official_val": 1, "official_test": 1},
        )

    def test_fraction_is_optional_alternative(self):
        records = normalize_records(
            [{"name": f"rain_{index}.jpg", "weather": "rain"} for index in range(20)]
        )
        self.assertEqual(len(select_subset(records, fraction=0.25, seed=42)), 5)

    def test_manifest_records_can_be_written_as_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            records = [{"image_path": "one.jpg", "split": "train", "conditions": ["rain"]}]
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])["image_path"],
                "one.jpg",
            )

    def test_configuration_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"source_domain": "cityscapes", "batch_size": 2}),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config["source_domain"], "cityscapes")
            self.assertEqual(config["batch_size"], 2)

    def test_cityscapes_annotation_matching_uses_unique_deterministic_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            annotation_root = root / "annotations"
            image_path = image_root / "train" / "city" / "example.png"
            annotation_path = annotation_root / "train" / "city" / "example.json"
            image_path.parent.mkdir(parents=True)
            annotation_path.parent.mkdir(parents=True)
            image_path.touch()
            annotation_path.touch()
            dataset = CityscapesDataset(image_root, annotation_root)
            self.assertEqual(dataset._annotation_for(image_path), annotation_path)

    def test_cityscapes_annotation_matching_reports_missing_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            annotation_root = root / "annotations"
            image_path = image_root / "example.png"
            image_root.mkdir()
            annotation_root.mkdir()
            image_path.touch()
            dataset = CityscapesDataset(image_root, annotation_root)
            with self.assertRaisesRegex(FileNotFoundError, "Verify annotation_root"):
                dataset._annotation_for(image_path)

    def test_cityscapes_annotation_matching_reports_ambiguous_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            annotation_root = root / "annotations"
            image_path = image_root / "example.png"
            image_root.mkdir()
            (annotation_root / "one").mkdir(parents=True)
            (annotation_root / "two").mkdir(parents=True)
            image_path.touch()
            (annotation_root / "one" / "example.json").touch()
            (annotation_root / "two" / "example.json").touch()
            dataset = CityscapesDataset(image_root, annotation_root)
            with self.assertRaisesRegex(RuntimeError, "Ambiguous annotations"):
                dataset._annotation_for(image_path)


if __name__ == "__main__":
    unittest.main()