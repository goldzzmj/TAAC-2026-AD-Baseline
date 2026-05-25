# Phase2-T4b: Label Smoothing 0.01

## 实验目的

测试轻微 Label Smoothing 对模型校准和泛化能力的影响。

## 唯一变量

vs T2 baseline (cosine LR + warmup, Eval AUC=0.8450):

- **新增**: `--label_smoothing 0.01`

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| label_smoothing | 0.01 | BCE loss 的标签平滑系数 |

## Label Smoothing 工作原理

在 BCE loss 计算前对标签进行平滑:
```
smooth_label = label * (1 - ls) + ls * 0.5
```
- label=1 → smooth = 1 - ls/2 = 0.995
- label=0 → smooth = ls/2 = 0.005

这可以防止模型过度自信，改善校准（logloss）。

## 文件列表

| 文件 | 来源 |
|------|------|
| train.py | T2 + EMA/LS argparse 参数（与 T3 相同） |
| trainer.py | T2 + EMA/LS 集成（与 T3 相同） |
| model.py | T2 + ExponentialMovingAverage 类（与 T3 相同） |
| dataset.py | COPY from T2 |
| utils.py | COPY from T2 |
| schema_utils.py | COPY from T2 |
| ns_groups.json | COPY from T2 |
| run.sh | T2 baseline + label_smoothing 参数 |
