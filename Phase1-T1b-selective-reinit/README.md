# Phase 1 - T1b: Selective Sparse Re-init

> Based on: Phase0 (v3 + time_split + deterministic seed)
> Experiment: T-1b from optimization plan

## Goal

Test the effect of **selective** sparse embedding re-initialization: only re-init embeddings with very high cardinality (>100k), starting from epoch 2.

## Only Change (vs Phase0 baseline)

| Parameter | Phase0 | T1b |
|-----------|--------|-----|
| reinit_sparse_after_epoch | 1 (every epoch) | **2 (delay to epoch 2)** |
| reinit_cardinality_threshold | 0 (all embeddings) | **100000 (only high-card)** |

## Hypothesis

- Low-cardinality embeddings (e.g., gender, age group) should NOT be reset - they learn stable representations quickly
- High-cardinality embeddings (e.g., user IDs with millions of values) may benefit from cold restart to reduce overfitting
- Delaying to epoch 2 gives embeddings one full epoch to learn before any reset

## Expected Results

- A middle ground between T1a (no re-init) and Phase0 (full re-init)
- May preserve low-card embedding quality while controlling high-card overfitting

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
| run.sh | NEW | Phase0 config + selective re-init |

### Model_Evaluation/ (inference code)
| File | Source |
|------|--------|
| infer.py | Phase0 |
| dataset.py | Phase0 |
| model.py | Phase0 |
