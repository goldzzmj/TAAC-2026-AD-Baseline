# Phase 1 - T2: Cosine LR Scheduler

> Based on: Phase0 (v3 + time_split + deterministic seed)
> Experiment: T-2 from optimization plan

## Goal

Test the effect of **cosine learning rate scheduling** with linear warmup.

## Only Change (vs Phase0 baseline)

| Parameter | Phase0 | T2 |
|-----------|--------|-----|
| lr_scheduler | none | **cosine** |
| warmup_steps | 0 | **500** |
| min_lr_ratio | 0.1 | 0.1 (unchanged) |

## Hypothesis

- Constant learning rate may lead to oscillation in later epochs
- Warmup prevents early instability with AdamW
- Cosine decay allows smoother convergence
- Note: LR scheduler only affects dense optimizer (AdamW), not sparse (Adagrad)

## Expected Results

- May improve convergence stability
- Could reduce train-test gap through better generalization
- Warmup=500 steps is conservative (~0.5 epoch at batch_size=64

## File List

### Model_Training/ (training code)
| File | Source | Change |
|------|--------|--------|
| dataset.py | Phase0 | No change |
| utils.py | Phase0 | No change |
| train.py | Phase0 | No change |
| trainer.py | Phase0 | No change |
| model.py | Phase0 | No change |
| schema_utils.py | Phase0 | No change |
| ns_groups.json | Phase0 | No change |
| run.sh | NEW | Phase0 config + cosine LR scheduler |

### Model_Evaluation/ (inference code)
| File | Source |
|------|--------|
| infer.py | Phase0 |
| dataset.py | Phase0 |
| model.py | Phase0 |
