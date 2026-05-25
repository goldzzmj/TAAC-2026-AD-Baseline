# Phase3-ACCEL: torch.compile Training Acceleration

## Experiment Purpose

Test whether `torch.compile(model, mode="reduce-overhead")` can speed up training
without degrading AUC. This is a pure engineering optimization - no model or
hyperparameter changes.

## Only Variable

vs T2 baseline (cosine LR, Eval AUC=0.8450):

- **Change**: `--use_compile` (wraps model with torch.compile)

## Parameter Comparison

| Parameter | T2 baseline | ACCEL-Compile |
|-----------|------------|---------------|
| torch.compile | Disabled | Enabled (reduce-overhead) |
| BF16 | Disabled | Disabled |

## Optimization Motivation

1. torch.compile fuses CUDA kernels, reducing kernel launch overhead
2. Expected 10-30% training speedup with no accuracy loss
3. Pure engineering change - no model architecture or hyperparameter differences
4. Code also includes `--use_bf16` flag for future BF16 experiments

## File List

| File | Source |
|------|--------|
| train.py | T2 + `--use_compile`, `--use_bf16` flags |
| trainer.py | T2 + BF16 autocast support |
| model.py | SAME as T2 |
| dataset.py | COPY from T2 |
| utils.py | COPY from T2 |
| schema_utils.py | COPY from T2 |
| ns_groups.json | COPY from T2 |
| run.sh | T2 baseline + `--use_compile` |
