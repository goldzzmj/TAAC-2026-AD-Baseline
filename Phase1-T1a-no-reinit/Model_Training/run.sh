#!/bin/bash
set -euo pipefail

# Phase1-T1a: Disable sparse re-init
# Only change vs Phase0 baseline:
#   reinit_sparse_after_epoch=-1 (disable re-init)
#
# All other params identical to Phase0 (v3 + time_split + deterministic seed).

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
    --run_name "${RUN_NAME:-phase1_t1a_no_reinit}" \
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
    --reinit_sparse_after_epoch -1 \
    "$@"
