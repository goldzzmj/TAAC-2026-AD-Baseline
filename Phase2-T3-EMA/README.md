# Phase2-T3: EMA (Exponential Moving Average)

## 实验目的

测试 EMA (指数移动平均) 对模型泛化能力的影响。

## 唯一变量

vs T2 baseline (cosine LR + warmup, Eval AUC=0.8450):

- **新增**: `--use_ema --ema_decay 0.999 --ema_warmup_steps 100`

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| use_ema | True | 启用 EMA |
| ema_decay | 0.999 | EMA 衰减因子 |
| ema_warmup_steps | 100 | EMA warmup 步数（线性增长 decay） |

## EMA 工作原理

1. 训练时维护模型参数的影子副本（shadow），存储在 CPU 上（节省 ~110MB GPU 显存）
2. 每步训练后更新 shadow: `shadow = shadow * decay + param * (1 - decay)`
3. 验证时使用 EMA shadow 参数替代训练参数
4. 验证结束后恢复训练参数

## 文件列表

| 文件 | 来源 |
|------|------|
| train.py | T2 + EMA/LS argparse 参数 |
| trainer.py | T2 + EMA/LS 集成 |
| model.py | T2 + ExponentialMovingAverage 类 |
| dataset.py | COPY from T2 |
| utils.py | COPY from T2 |
| schema_utils.py | COPY from T2 |
| ns_groups.json | COPY from T2 |
| run.sh | T2 baseline + EMA 参数 |
