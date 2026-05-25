#!/bin/bash
set -euo pipefail

# Phase3-ACCEL-BF16: BF16 mixed precision training acceleration
# Only change vs T2 baseline (cosine LR):
#   --use_bf16  (torch.autocast with bfloat16)
#
# Goal: confirm BF16 does NOT degrade AUC while speeding up training.
# Unlike torch.compile, BF16 is purely a precision change — no graph compilation.
# All other params identical to T2 baseline.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/outputs/cloud_run"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${DEFAULT_OUTPUT_ROOT}/checkpoints}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${DEFAULT_OUTPUT_ROOT}/logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${DEFAULT_OUTPUT_ROOT}/tensorboard}"

mkdir -p "${TRAIN_CKPT_PATH}" "${TRAIN_LOG_PATH}" "${TRAIN_TF_EVENTS_PATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN_CMD=("${PYTHON_BIN}")
else
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN_CMD=("python3")
    else
        PYTHON_BIN_CMD=("python")
    fi
fi

"${PYTHON_BIN_CMD[@]}" -u "${SCRIPT_DIR}/train.py" \
    --run_name "${RUN_NAME:-phase3_accel_bf16}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "__disabled__" \
    --seq_encoder_type transformer \
    --seq_top_k 50 \
    --num_hyformer_blocks 2 \
    --dropout_rate 0.01 \
    --emb_skip_threshold 1000000 \
    --emb_hash_size 0 \
    --target_attention_hidden_mult 3 \
    --cross_fusion_layers 1 \
    --cross_fusion_rank 32 \
    --context_fusion_hidden_mult 2 \
    --fusion_gate_init 0.1 \
    --num_workers 8 \
    --batch_size 64 \
    --num_epochs 8 \
    --patience 3 \
    --valid_ratio 0.05 \
    --time_split \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dense_weight_decay 0.01 \
    --lr_scheduler cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.1 \
    --use_bf16 \
    "$@"
