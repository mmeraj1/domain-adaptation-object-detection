# Dataset preparation and layout

This directory is reserved for local datasets, manifests, and dataset metadata.
The repository does **not** download or redistribute any dataset. Dataset files
are ignored by Git.

## Roles and purpose

- **Cityscapes** is the source domain for clear urban-scene object detection.
- **Foggy Cityscapes** is target domain 1, used to study adverse-weather domain
  shift from clear to foggy imagery.
- **BDD100K adverse-weather data** is target domain 2. Development uses a
  deterministic, manifest-only approximately 10% subset covering the rain,
  snow, and night metadata dimensions. The full dataset is reserved for later
  experiments.

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
4. Store the generated manifest under `data/processed/bdd100k_subset/`.
	Images and original annotations remain in their original locations; they
	are not copied or altered.
5. Run `scripts/check_dataset.py` against each configured domain before any
	model work.

The subset script accepts metadata records containing an image path/name and
weather/time-of-day fields. Since BDD100K metadata schemas can vary by release
or local preprocessing, the exact field names and annotation linkage must be
verified from the acquired files. Unrecognized records with an image path are
reported as unclassified and excluded from the adverse-condition subset.

## Subset and reproducibility

The default target is `10,000` images (`--target-count 10000`) and the default
seed is `42`. `--fraction` is available as an optional alternative based on
the number of adverse-condition candidates. Sampling uses all three condition
dimensions jointly and minimizes deviation from equal rain, snow, and night
counts. Only records classified into at least one dimension are eligible. A
record may belong to multiple buckets, contributes to each applicable count,
and is selected at most once, so conditions are not treated as mutually
exclusive. Existing source split fields are preserved; missing split fields
remain unspecified and no research split is silently created.
The manifest preserves the source metadata and records the assigned split and
conditions. The adjacent `.stats.json` file records counts, seed, and fraction.

For reproducibility, retain the dataset release/version, metadata file version,
manifest, seed, split ratios, configuration revision, and preprocessing
statistics with the experiment record.