# TAAC 2026 AD Baseline — 系统化消融实验

PyTorch research workspace for TAAC 2026 advertising recommendation experiments, with systematic control-variable ablation studies.

## 项目概述

本项目基于 [TAAC2026 腾讯广告算法竞赛](https://taac.qq.com/)，在 OneTrans-small baseline 基础上，按照 `TAAC2026_优化规划.md` 进行了系统性的控制变量消融实验，覆盖训练策略、正则化、模型结构、特征工程、训练加速等多个优化方向。

### 核心发现

- **最佳方案**: Phase1-T2-cosine-lr, 推理 AUC = **0.844955** (+0.61% vs baseline)
- **原始基线**: v3 HyFormer, 推理 AUC = 0.839805
- **关键结论**: Cosine LR scheduler 对推理泛化有显著帮助，即使训练 AUC 未明显提升

## 实验结果汇总

| 排名 | Phase | 推理 AUC | 方向 | 状态 |
|------|-------|----------|------|------|
| 1 | Phase1-T2-cosine-lr | **0.844955** | 训练策略 | Best |
| 2 | Phase2-T6a-FeatMask (ep4) | 0.844911 | 正则化 | 接近 |
| 3 | Phase2-T6a-FeatMask (ep5) | 0.844416 | 正则化 | 接近 |
| 4 | Phase5.5-QRY1-TargetAwareQuery (ep5) | 0.844115 | 架构优化 | 接近 |
| 5 | Phase4-U-D1-DenseProj-v2 | 0.843732 | 特征工程 | 轻负向 |
| 6 | Phase3-ACCEL-BF16-v2 (ep5) | 0.84327 | 训练加速 | 负向 |
| 7 | Phase5.5-DOM1-DomainGate-v2 (ep4) | 0.843100 | 架构优化 | 负向 |
| 8 | Phase5.5-TEMP1-TemporalGate (ep5) | 0.843035 | 架构优化 | 负向 |
| 9 | Phase4-SEQ2a-TimePeriod (ep4) | 0.842944 | 特征工程 | 负向 |
| 10 | Phase2-T5-Dropout | 0.842873 | 正则化 | 负向 |
| - | Phase0 (baseline verify) | 0.838662 | 验证修正 | baseline |

> 完整 AUC 排行榜和详细分析见 `TAAC2026_优化规划.md`

## 目录结构

```
TAAC-2026-AD-Baseline/
├── OneTrans_pytorch/          # 原始 OneTrans 架构复现
├── src/                       # 原始 baseline 代码
├── run/                       # 原始运行脚本
├── requirements.txt           # Python 依赖
├── TAAC2026_优化规划.md        # 优化规划文档（核心）
│
├── Phase0-验证修正与种子固定/   # 基线验证 + 时间切分 + 种子固定
├── Phase1-T1a-no-reinit/      # 消融: 无权重重初始化
├── Phase1-T1b-selective-reinit/ # 消融: 选择性重初始化
├── Phase1-T2-cosine-lr/       # 消融: Cosine LR (最佳)
├── Phase2-T3-EMA/             # 消融: EMA 0.999
├── Phase2-T4b-LabelSmoothing/ # 消融: Label Smoothing 0.01
├── Phase2-T4b-LS/             # 消融: Label Smoothing (var)
├── Phase2-T5-Dropout/         # 消融: Dropout 0.02
├── Phase2-T6a-FeatMask/       # 消融: Feature Masking 0.05
├── Phase2-T9-long/            # 消融: 更长训练
├── Phase3-ACCEL-BF16/         # 加速: BF16 混合精度
├── Phase3-ACCEL-Compile/      # 加速: torch.compile
├── Phase3.5-GAP4-SWA/         # 策略: SWA 随机权重平均
├── Phase4-SEQ1-fid47Treatment/ # 特征: 移除 seq_c fid47
├── Phase4-SEQ2a-TimePeriod/   # 特征: 时间周期编码
├── Phase4-U-D1-DenseProj/     # 特征: Dense MLP 投影
├── Phase4-U-D2-SharedFID-JointModeling/ # 特征: Shared-FID 联合建模
├── Phase5-DIN1-DINQueryEnhancement/ # 结构: DIN Query 增强
├── Phase5-M2-HyFormerBlocks1/ # 结构: HyFormer 1 block
├── Phase5-M3-HyFormerBlocks3/ # 结构: HyFormer 3 blocks
├── Phase5-MLP1-DeepClassifier/ # 结构: Classifier 加深
├── Phase5.5-DOM1-DomainGate/  # 架构: 域级动态门控
├── Phase5.5-QRY1-TargetAwareQuery/ # 架构: Target-Aware Query
├── Phase5.5-TEMP1-TemporalGate/ # 架构: 时间条件化门控
├── Phase6-ACCEL-3-GradientCheckpointing/ # 加速: 梯度检查点
├── Phase-ENS1-MultiSeedEnsemble/ # 集成: 多种子 Ensemble
├── Phase-IT1-IDFeatureTreatment/ # 特征: ID 特征处理
└── Phase-OPT1-MuonOptimizer/  # 优化器: Muon
```

每个 Phase 目录下包含：
- `README.md` — 该 Phase 的实验说明和结果
- `Model_Training/` — 训练代码（model.py, train.py, trainer.py, dataset.py, run.sh 等）
- `Model_Evaluation/` — 推理评估代码（model.py, infer.py, dataset.py）
- `submit_bundle/` — 提交打包文件（部分 Phase 包含）

## 原始 Baseline

### Included Projects

#### 1. Demo baseline scaffold

Root project contains a compact OneTrans-small style baseline for the TAAC2026 demo parquet dataset.

- `run/pipeline/training/train_onetrans_small.py`: training entry point
- `src/data_module/dataset/taac2026_demo_dataset.py`: parquet reader and collate
- `src/model_module/model/onetrans_small.py`: unified tokenization baseline model
- `src/trainer_module/onetrans_trainer.py`: training and evaluation loop
- `src/utils/metrics.py`: accuracy and AUC helpers

#### 2. OneTrans architecture reproduction

`OneTrans_pytorch/` is a standalone PyTorch reproduction of the core model architecture described in the paper:

- paper: `https://arxiv.org/html/2510.26104v3`
- unified non-sequential and sequential tokenization
- mixed shared/token-specific causal attention
- mixed shared/token-specific FFN
- pyramid token pruning stack

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run the compact baseline on local parquet files:

```bash
python run/pipeline/training/train_onetrans_small.py \
    --train-path data/demo_1000_train.parquet \
    --valid-path data/demo_1000_test.parquet \
    --output-dir outputs/onetrans_small_demo
```

Run the standalone OneTrans architecture forward demo:

```bash
python OneTrans_pytorch/run/pipeline/training/demo_forward.py
```

## Notes

- Data directory (`data/`) is excluded from Git. Place local parquet files under `data/`.
- The root baseline currently treats `label_type` as a two-class label and uses the largest raw value as the positive class by default.
- `label_time` is excluded from the root baseline input to reduce future leakage risk.
