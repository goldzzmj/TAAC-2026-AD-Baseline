# Phase 0: 验证修正与种子固定

> 对应优化规划文档: `TAAC2026_优化规划.md` → Phase 0
> 实验编号: V-1 (时间切分验证) + V-2 (随机种子固定)

## 目标

在开始所有消融实验之前，修正两个基础设施问题：

1. **时间切分验证 (V-1)**: 当前验证集按 Row Group 文件顺序尾部切分，不保证是最近的数据。改为按 timestamp 排序后取最新的 Row Groups 作为验证集，使 valid AUC 更接近真实推理 AUC。
2. **确定性种子 (V-2)**: 当前缺少 `cudnn.benchmark=False`、`torch.use_deterministic_algorithms(True)` 等，导致多次运行结果不一致。补全所有确定性设置。

## 唯一变更 (相对 v3 baseline)

### 变更 1: dataset.py — `get_pcvr_data()` 时间切分

- 新增 `_get_rg_timestamp_stats()` 函数: 利用 Parquet 元数据获取每个 Row Group 的 timestamp max
- 新增 `time_split` 参数 (默认 True): 按 timestamp 排序 Row Groups，时间最新的作为验证集
- 新增 `row_group_list` 参数: 支持传入预排序的 Row Group 列表

### 变更 2: utils.py — `set_seed()` 完全确定性

- `torch.backends.cudnn.benchmark = False`
- `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- 新增 `full_deterministic` 参数 (默认 True)

### 变更 3: train.py — 添加参数

- 新增 `--time_split` / `--no_time_split` 参数，传递给 `get_pcvr_data()`

## 不变的参数

所有训练超参数与 v3 baseline 完全一致:
```
batch_size=64, lr=1e-4, sparse_lr=0.05, dropout=0.01
rankmixer(full), user_ns=5, item_ns=2, num_queries=2
transformer encoder, seq_top_k=50, num_blocks=2
无 EMA, 无 scheduler, 无 label_smoothing
valid_ratio=0.05, patience=3, num_epochs=8
```

## 预期结果

- Valid AUC 可能下降 (因为时间切分更严格)
- Valid→Inference Gap 应该缩小 (从 2.38% 降低)
- 多次运行的 AUC 方差应显著缩小

## 文件列表

### Model_Training/ (训练代码)
| 文件 | 来源 | 改动 |
|------|------|------|
| dataset.py | v3 + 修改 | +time_split, +row_group_list, +_get_rg_timestamp_stats |
| utils.py | v3 + 修改 | set_seed() 增加完全确定性 |
| train.py | v3 + 修改 | +--time_split 参数 |
| trainer.py | v3 原样 | 无改动 |
| model.py | v3 原样 | 无改动 |
| schema_utils.py | v3 原样 | 无改动 |
| ns_groups.json | v3 原样 | 无改动 |
| run_compact.sh | 新建 | v3 配置 + --time_split |

### Model_Evaluation/ (推理代码)
| 文件 | 来源 |
|------|------|
| infer.py | v3 原样 |
| dataset.py | v3 原样 |
| model.py | v3 原样 |
