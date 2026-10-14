# Cityscapes → Foggy Cityscapes: Experiment Log

Date: 9 October 2026

## Research objective

Reproduce source-only and Adaptive Teacher baselines for unsupervised domain adaptation in object detection.

Source: Cityscapes (labeled)
Target: Foggy Cityscapes (unlabeled during training)

Planned analysis:
- Detection performance across fog densities
- Source-only versus Adaptive Teacher
- Reliability of teacher-generated pseudo-labels

## Environment

- Platform: RunPod
- GPU: NVIDIA A40
- PyTorch: 1.7
- CUDA: 11
- Detectron2: v0.3
- Adaptive Teacher repository commit: 5256463
- Dataset: Cityscapes → Foggy Cityscapes
- Backbone: VGG16
- Classes: 8

## Dataset preparation

- Cityscapes source training images: 2,975
- Usable labeled training images reported by Detectron2: 2,965
- Foggy Cityscapes training images: 8,925
- Foggy Cityscapes validation images: 1,500
- Fog levels: beta = 0.005, 0.01, 0.02
- Target training labels are not used for adaptation.

## Source-only smoke tests

### Attempt 1: Missing supervision seed file

Error:
FileNotFoundError: dataseed/COCO_supervision.txt

Resolution:
Created a Cityscapes-specific supervision seed JSON and configured
DATALOADER.RANDOM_DATA_SEED_PATH.

### Attempt 2: Unsupported learning-rate scheduler

Error:
ValueError: Unknown LR scheduler: WarmupTwoStageMultiStepLR

Resolution:
Added a build_lr_scheduler override to BaselineTrainer to use the
Adaptive Teacher scheduler implementation.

Patch:
reproducibility/source_only_scheduler_fix.patch

### Attempt 3: Protobuf incompatibility

Error:
TypeError: Descriptors cannot be created directly

Resolution:
Installed protobuf==3.20.3.

### Attempt 4: Successful smoke test

- Iterations: 10
- Training completed successfully
- Foggy validation evaluation completed
- AP: approximately 0.0003%
- Final checkpoint saved

## Source-only 100-iteration benchmark

Configuration:
reproducibility/configs/source_only_benchmark.yaml

Training command:
python train_net.py --num-gpus 1 --config-file experiments/cityscapes_to_foggy/source_only/source_only_benchmark.yaml

Settings:
- Batch size: 2
- Base learning rate: 0.005
- Maximum iterations: 100
- Backbone initialized randomly
- Training time: approximately 48 seconds
- Maximum GPU memory: approximately 4,330 MB

Results:
- AP: 0.0003%
- AP50: 0.0007%
- AP75: 0.0002%

Checkpoint:
source_only/benchmark/model_final.pth

Training log:
source_only/benchmark_100.log

Interpretation:
This is a pipeline and runtime benchmark, not a scientifically
meaningful source-only detection baseline.

## Resume test: 100 → 120 iterations

Configuration:
reproducibility/configs/source_only_resume_test.yaml

Command:
python train_net.py --num-gpus 1 --config-file experiments/cityscapes_to_foggy/source_only/source_only_resume_test.yaml --resume

Verified output:
- Model checkpoint loaded
- Optimizer state loaded
- Scheduler state loaded
- Starting training from iteration 100
- Final iteration: 119
- Final loss: 0.5553
- Learning rate: 0.00059941
- Process exit status: 0

Evaluation:
- AP: 0.0000%
- AP50: 0.0002%
- AP75: 0.0000%

Conclusion:
Checkpoint continuation works correctly.

## Official Adaptive Teacher configuration

- Trainer: ateacher
- Maximum iterations: 100,000
- Base learning rate: 0.04
- Burn-in: 20,000 iterations
- Pseudo-label threshold: 0.8
- EMA keep rate: 0.9996
- Original training instructions: 8 GPUs

The single-GPU source-only benchmark uses a smaller batch size
and learning rate. These are reproduction choices.

## Remaining work

1. Finalize the full source-only training schedule and budget.
2. Train and evaluate the source-only baseline.
3. Train and evaluate Adaptive Teacher.
4. Evaluate AP separately for each fog density.
5. Analyze pseudo-label reliability.
6. Commit experiment configurations, code changes, and results to GitHub.

## Reproducibility notes

Original terminal logs, configurations, checkpoints, and scheduler
patch are preserved under:
/workspace/experiments/cityscapes_to_foggy/

The full source-only and Adaptive Teacher baselines have NOT yet
been completed.
