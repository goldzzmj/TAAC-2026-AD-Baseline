# Phase2-T5: Dropout 0.02

## 实验目的

测试增加 Dropout 对模型泛化能力的影响。

## 唯一变量

vs T2 baseline (cosine LR + warmup, Eval AUC=0.8450):

- **改动**: `--dropout_rate 0.02` (从 0.01 提升到 0.02)

## 参数说明

| 参数 | T2 baseline | T5 | 说明 |
|------|------------|-----|------|
| dropout_rate | 0.01 | 0.02 | 2x dropout, 温和正则化 |

## 优化动机

1. 当前 train-valid gap ~2% (Valid AUC=0.8658 vs Eval AUC=0.8450)
2. Dropout 0.01 是一个非常低的值，微增到 0.02 可能帮助泛化
3. v4 中 dropout 0.05 被证明过强，但 0.02 是更温和的中间值
4. 无需任何代码修改，仅改超参

## 文件列表

| 文件 | 来源 |
|------|------|
| train.py | SAME as T3 (含 EMA/LS/FeatMask 支持，但默认关闭) |
| trainer.py | SAME as T3 |
| model.py | SAME as T3 |
| dataset.py | COPY from T2 |
| utils.py | COPY from T2 |
| schema_utils.py | COPY from T2 |
| ns_groups.json | COPY from T2 |
| run.sh | T2 baseline + dropout_rate=0.02 |
