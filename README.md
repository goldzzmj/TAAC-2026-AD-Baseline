# TAAC 2026 AD Baseline

PyTorch research workspace for TAAC 2026 advertising recommendation experiments.

## Included Projects

### 1. Demo baseline scaffold

This root project contains a compact OneTrans-small style baseline for the
TAAC2026 demo parquet dataset.

- `run/pipeline/training/train_onetrans_small.py`: training entry point
- `src/data_module/dataset/taac2026_demo_dataset.py`: parquet reader and collate
- `src/model_module/model/onetrans_small.py`: unified tokenization baseline model
- `src/trainer_module/onetrans_trainer.py`: training and evaluation loop
- `src/utils/metrics.py`: accuracy and AUC helpers

### 2. OneTrans architecture reproduction

`OneTrans_pytorch/` is a standalone PyTorch reproduction of the core model
architecture described in the paper:

- paper: `https://arxiv.org/html/2510.26104v3`
- unified non-sequential and sequential tokenization
- mixed shared/token-specific causal attention
- mixed shared/token-specific FFN
- pyramid token pruning stack

Key files:

- `OneTrans_pytorch/src/model_module/model/schema.py`
- `OneTrans_pytorch/src/model_module/model/tokenizer.py`
- `OneTrans_pytorch/src/model_module/model/layers.py`
- `OneTrans_pytorch/src/model_module/model/onetrans.py`
- `OneTrans_pytorch/run/pipeline/training/demo_forward.py`

## Local Data

The `data/` directory is intentionally excluded from Git.

- local parquet files should be placed under `data/`
- dataset files are not uploaded to GitHub
- generated outputs should be written to `outputs/`

## Install

Root baseline:

```bash
pip install -r requirements.txt
```

OneTrans reproduction:

```bash
pip install -r OneTrans_pytorch/requirements.txt
```

## Quick Start

Run the compact baseline on local parquet files:

```bash
python run/pipeline/training/train_onetrans_small.py --train-path data/demo_1000_train.parquet --valid-path data/demo_1000_test.parquet --output-dir outputs/onetrans_small_demo
```

Run the standalone OneTrans architecture forward demo:

```bash
python OneTrans_pytorch/run/pipeline/training/demo_forward.py
```

## Notes

- The root baseline currently treats `label_type` as a two-class label and uses
  the largest raw value as the positive class by default.
- `label_time` is excluded from the root baseline input to reduce future
  leakage risk.
- The OneTrans reproduction focuses on the model architecture rather than full
  production serving optimizations.
