# Phase 1 - T1a: Disable Sparse Re-init

> Based on: Phase0 (v3 + time_split + deterministic seed)
> Experiment: T-1a from optimization plan

## Goal

Test the effect of **completely disabling** sparse embedding re-initialization.

## Only Change (vs Phase0 baseline)

| Parameter | Phase0 | T1a |
|-----------|--------|-----|
| reinit_sparse_after_epoch | 1 (default, full re-init every epoch) | **-1 (disabled)** |

## Hypothesis

The default re-init strategy (epoch=1, threshold=0) resets ALL embeddings every epoch, which may be too aggressive and destroy learned representations. Disabling it should allow embeddings to accumulate knowledge across epochs.

## Expected Results

- If re-init hurts: Valid AUC may improve, train-test gap may shrink
- If re-init helps: Valid AUC may drop (overfitting on embeddings)

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
| run.sh | NEW | Phase0 config + --reinit_sparse_after_epoch -1 |

### Model_Evaluation/ (inference code)
| File | Source |
|------|--------|
| infer.py | Phase0 |
| dataset.py | Phase0 |
| model.py | Phase0 |
