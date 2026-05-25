# Phase2-T6a: Feature Masking 0.05

## 实验目的

测试 Feature Masking (特征随机遮蔽) 对模型泛化能力的影响。

## 唯一变量

vs T2 baseline (cosine LR + warmup, Eval AUC=0.8450):

- **新增**: `--feature_mask_ratio 0.05`

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| feature_mask_ratio | 0.05 | 训练时以 5% 概率随机将 dense feature 维度置零 |

## Feature Masking 工作原理

1. 仅在训练阶段生效（`self.model.training` 为 True 时）
2. 对每个 batch 的 dense features 生成随机 mask 矩阵
3. 以 `feature_mask_ratio` 概率将各维度置零
4. 验证和推理阶段不应用 masking
5. 目的：作为正则化手段，防止模型过度依赖特定特征维度

## 背景与动机

v6 (第六次-HyFormer) 同时引入了 EMA + Label Smoothing + Feature Masking + Cosine LR，
但 Valid AUC=0.8615 低于 Phase0 baseline 的 0.8655。

本实验通过控制变量法，单独测试 Feature Masking 的效果，确认其是正向还是负向贡献。

## 文件列表

| 文件 | 来源 |
|------|------|
| train.py | T3 + feature_mask_ratio argparse |
| trainer.py | T3 + _apply_feature_masking 方法 |
| model.py | T2 + ExponentialMovingAverage 类 |
| dataset.py | COPY from T2 |
| utils.py | COPY from T2 |
| schema_utils.py | COPY from T2 |
| ns_groups.json | COPY from T2 |
| run.sh | T2 baseline + feature_mask_ratio 0.05 |
