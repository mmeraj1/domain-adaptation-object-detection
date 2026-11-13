# Experiment 10 — Fixed-teacher pseudo-label agreement: failure audit

Generated: 2026-11-13T16:30:45

## Question and protocol
Frozen teacher on strong-fog validation images. Original-vs-flip 50 scenes; independent weak photometric views 20 scenes. Per-image matched K fixed **before GT evaluation**. Class-aware greedy IoU ≥ 0.5 matching. Crowd annotations excluded. These metrics are diagnostic P/R/F1, not mAP.

## 50-scene original/flip comparison
| experiment       | method               |   retained |   gt_objects |   TP |   FP |   FN |   precision |   recall |     F1 |
|:-----------------|:---------------------|-----------:|-------------:|-----:|-----:|-----:|------------:|---------:|-------:|
| original_flip_50 | confidence_agreement |       1351 |          749 |  357 |  994 |  392 |      0.2642 |   0.4766 | 0.3400 |
| original_flip_50 | confidence_only      |       1351 |          749 |  385 |  966 |  364 |      0.2850 |   0.5140 | 0.3667 |

## 20-scene independent photometric comparison
| experiment                 | method               |   retained |   gt_objects |   TP |   FP |   FN |   precision |   recall |     F1 |
|:---------------------------|:---------------------|-----------:|-------------:|-----:|-----:|-----:|------------:|---------:|-------:|
| independent_photometric_20 | confidence_agreement |       1258 |          310 |  169 | 1089 |  141 |      0.1343 |   0.5452 | 0.2156 |
| independent_photometric_20 | confidence_only      |       1258 |          310 |  173 | 1085 |  137 |      0.1375 |   0.5581 | 0.2207 |

## 0.70–0.80 selected-box evidence
| experiment                 | method               | band      |   selected |   TP |   FP |   selected_precision |
|:---------------------------|:---------------------|:----------|-----------:|-----:|-----:|---------------------:|
| independent_photometric_20 | confidence_agreement | 0.70–0.80 |         45 |    9 |   36 |               0.2000 |
| independent_photometric_20 | confidence_only      | 0.70–0.80 |         50 |   11 |   39 |               0.2200 |
| original_flip_50           | confidence_agreement | 0.70–0.80 |         66 |   24 |   42 |               0.3636 |
| original_flip_50           | confidence_only      | 0.70–0.80 |        113 |   32 |   81 |               0.2832 |

## Scene-level uncertainty (exploratory)
| experiment                 |   n_scenes |   precision_delta_pp |   bootstrap_low_pp |   bootstrap_high_pp |
|:---------------------------|-----------:|---------------------:|-------------------:|--------------------:|
| independent_photometric_20 |         20 |              -0.3180 |            -0.7502 |              0.0843 |
| original_flip_50           |         50 |              -2.0725 |            -3.3577 |             -0.7993 |

## Selection swaps
| experiment                 |    K |   shared |   swapped_pairs |
|:---------------------------|-----:|---------:|----------------:|
| independent_photometric_20 | 1258 |     1164 |              94 |
| original_flip_50           | 1351 |      776 |             575 |

## Interpretation and limitations
- The existing 50-scene comparison is a **negative result** for the tested confidence×agreement ranking; no improvement is claimed.
- The 0.70–0.80 results describe **selected boxes**, not all candidates in that band. Confidence-only and agreement-selected subsets differ, and per-set GT matching is set-dependent.
- Ground truth was consulted only **after** the original selection files were produced; this notebook evaluates those saved selections.
- These are small, nonrandom scene subsets, not repeated-seed training studies. No statistical significance or generalization claim follows from bootstrap intervals alone.
- Crowd ground truth is excluded; this is not full COCO crowd-ignore evaluation.
- No matched downstream training, fog-level ablation, or second-dataset validation is performed here.
- Historical training logs verify recorded configuration and execution but not exact historical target sample identities.

## Next experiment (not yet completed)
Inspect the complete View-A candidate pool, particularly 0.70–0.80 scores, for stable **true positives versus stable false positives**. Define a refinement **without using validation GT for training-time selection**, then run equal-budget baseline vs refined training with matched seed/config, multiple seeds and another fog dataset. Report target mAP@0.5 and full COCO AP as appropriate.

## Harmonious Teacher (CVPR 2023)
Related prior work addresses pseudo-label classification/localization harmony. This experiment is frozen-teacher, two-view class/box agreement with matched-count ranking; it is **not** a Harmonious Teacher reproduction. Novelty and comparisons must be verified against the paper.

## Artifact provenance
See `input_manifest.json` for SHA-256 of the three source files. Output CSV and PNG files in this folder are derived diagnostic evidence.
