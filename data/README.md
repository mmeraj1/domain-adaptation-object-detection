# Dataset preparation and layout

This directory is reserved for local datasets, manifests, and dataset metadata.
The repository does **not** download or redistribute any dataset. Dataset files
are ignored by Git.

## Roles and purpose

- **Cityscapes** is the source domain for clear urban-scene object detection.
- **Foggy Cityscapes** is target domain 1, used to study adverse-weather domain
  shift from clear to foggy imagery.
- **BDD100K** is target domain 2 for adverse-driving conditions. The validated
	local development subset contains 10,000 unique images focused on rain, snow,
	and night. Condition counts overlap: an image can be both rainy and night,
	or snowy and night, and is still stored only once.

## Expected local layout

Create or adapt the following layout after obtaining the datasets through their
official access procedures. The exact archive contents, directory names, and
annotation locations must be **verified against the downloaded release**; no
unverified URL, checksum, or machine-specific path is assumed here.

```text
data/
├── raw/
│   ├── cityscapes/
│   ├── foggy_cityscapes/
│   └── bdd100k/
└── processed/
	 └── bdd100k_subset/
```

The YAML configuration supplies explicit `root`, `annotation_root`,
`images_root`, `labels_path`, and manifest paths. For Cityscapes and Foggy
Cityscapes, configure the image directory and the directory containing the
Cityscapes-style polygon JSON annotations separately. Confirm the actual
relative layout and file naming after extraction. For BDD100K, confirm the
image location and the metadata/label format from the acquired release before
running preprocessing.

## Obtaining data

Obtain Cityscapes, Foggy Cityscapes, and BDD100K through the applicable
official project access/licensing process. Record the release/version and
access date in your experiment notes. Do not place credentials, downloaded
archives, or dataset images in this repository.

## Preprocessing procedure

1. Place each locally obtained dataset under `data/raw/` or point the config
	at an equivalent external location.
2. Verify image and annotation locations and update
	`configs/baseline_config.yaml` with local **relative** paths or paths
	supplied outside version control.
3. For BDD100K, provide a metadata JSON/JSONL file to
	`scripts/create_bdd100k_subset.py`. The script extracts rain, snow, and
	night as independent dimensions, selects approximately 10,000 unique
	adverse-condition images with approximately equal condition representation,
	and writes only a manifest.
4. Store the generated manifest under `data/processed/bdd100k_subset/` and
   selected images under `data/raw/bdd100k/images/train/` and
   `data/raw/bdd100k/images/val/`.
5. Run `scripts/check_dataset.py` against each configured domain before any
   model work.

The subset script accepts metadata records containing an image path/name and
weather/time-of-day fields. Since BDD100K metadata schemas can vary by release
or local preprocessing, the exact field names and annotation linkage must be
verified from the acquired files. Unrecognized records with an image path are
reported as unclassified and excluded from the adverse-condition subset.

## BDD100K

### Validated subset

The current development subset was generated with seed `42` and contains:

| Measure | Count |
| --- | ---: |
| Total unique images | 10,000 |
| Original `train` split | 8,761 |
| Original `val` split | 1,239 |
| Original `test` split | 0 |
| Rain | 4,852 |
| Snow | 4,851 |
| Night | 4,851 |
| Rain + night | 2,300 |
| Snow + night | 2,254 |
| Rain + snow | 0 |
| Duplicate selected images | 0 |
| Missing selected images | 0 |
| Invalid boxes | 0 |
| Total boxes/labels | 112,642 |

The subset preserves the original BDD100K `train` and `val` split metadata. No
synthetic test split was created and no test annotations were fabricated.

### Metadata and class normalization

BDD100K metadata is normalized to the research conditions:

- `rainy` -> `rain`
- `snowy` -> `snow`
- `night` -> `night`

Detection classes are normalized without mapping unrelated categories:

- `bike` -> `bicycle`
- `motor` -> `motorcycle`

The canonical detection classes and IDs are unchanged:

```text
person, rider, car, truck, bus, train, motorcycle, bicycle
```

### Current local structure

```text
data/
├── raw/
│   └── bdd100k/
│       └── images/
│           ├── train/
│           └── val/
└── processed/
		└── bdd100k_subset/
				├── manifest.jsonl
				└── manifest.jsonl.stats.json
```

`manifest.jsonl` contains the selected image names, original split,
normalized conditions, canonical labels, boxes, and source metadata.
`manifest.jsonl.stats.json` records the selection and validation statistics.

### Git and reproducibility policy

- BDD100K JPG/PNG image files are local-only and are not committed to Git.
- The generated 74 MB `manifest.jsonl` is ignored by Git.
- The small `manifest.jsonl.stats.json` file is version-controlled for
	reproducibility and audit information.
- Dataset archives and full raw BDD100K releases are not committed.

### Regenerate the subset

The original BDD100K images and both annotation files must already be available
locally. The script accepts one or more metadata files and requires an image
root, subset image root, and manifest output path. For the verified BDD100K
release, run:

```bash
python scripts/create_bdd100k_subset.py \
	--metadata /path/to/bdd100k_labels_images_train.json \
						/path/to/bdd100k_labels_images_val.json \
	--images-root /path/to/bdd100k/images/100k \
	--subset-root data/raw/bdd100k/images \
	--output data/processed/bdd100k_subset/manifest.jsonl \
	--target-count 10000 \
	--seed 42
```

The command uses only supported options. It preserves source train/val splits,
copies only selected images, and does not modify the original dataset.

### Validation commands

Run the tests:

```bash
python -m pytest tests/test_data_pipeline.py -v
```

Validate the generated BDD train and validation subsets:

```bash
python scripts/check_dataset.py \
	--config configs/baseline_config.yaml \
	--role target \
	--split train \
	--max-samples 3

python scripts/check_dataset.py \
	--config configs/baseline_config.yaml \
	--role target \
	--split val \
	--max-samples 3
```

## Status

- **BDD100K pipeline status:** validated.
- **Cityscapes and Foggy Cityscapes:** pending dataset access/validation.