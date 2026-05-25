# Phase2-T9: Longer Training

## 实验目的

测试更长训练周期（更多 epoch + 更大 patience）是否能提升模型性能。

## 唯一变量

vs T2 baseline (cosine LR + warmup, Eval AUC=0.8450):

- **修改**: `--num_epochs 12 --patience 5`（原为 8 和 3）

## 参数说明

| 参数 | T2 值 | T9 值 | 说明 |
|------|--------|--------|------|
| num_epochs | 8 | 12 | 最大训练 epoch 数 |
| patience | 3 | 5 | Early stopping patience |

## 注意

Cosine LR scheduler 的 total_steps 会随 num_epochs 增加，使衰减曲线更平滑。
这本身不是独立变量，而是更长训练的自然结果。

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
| run.sh | T2 baseline + 覆盖 num_epochs/patience |
