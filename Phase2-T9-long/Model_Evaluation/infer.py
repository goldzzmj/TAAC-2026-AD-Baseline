"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the structural defaults used by the third-run launcher).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory or a direct ``*.pt`` file.
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os

# These must be set before importing pyarrow/pandas/torch worker processes.
# The competition machines expose many CPU cores; without caps, each DataLoader
# worker may spawn large CPU thread pools and flood logs with NumExpr messages.
os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These match the structural defaults used by third-run ``run.sh``. When
# train_config.json is present it remains the single source of truth.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 2,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 1000000,
    'emb_hash_size': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 5,
    'item_ns_tokens': 2,
    'use_target_attention': True,
    'target_attention_hidden_mult': 3,
    'use_cross_fusion': True,
    'cross_fusion_layers': 1,
    'cross_fusion_rank': 32,
    'use_context_fusion': True,
    'context_fusion_hidden_mult': 2,
    'fusion_gate_init': 0.1,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 8
_FALLBACK_MIN_BATCH_SIZE = 4


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        logging.warning("Ignoring invalid integer env %s=%r; using %s", name, raw, default)
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def resolve_eval_batch_size(train_config: Dict[str, Any]) -> Tuple[int, int]:
    """Resolve inference batch size independently from training batch size.

    Inference uses a smaller default batch_size than training to avoid OOM on
    shared GPU environments. The OOM retry loop can reduce down to min_batch_size.
    EVAL_BATCH_SIZE / INFER_BATCH_SIZE env vars can override when needed.
    """
    train_bs = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    # Use a smaller default for inference: cap at 32 to avoid OOM on shared GPUs
    default_eval_bs = min(train_bs, 32)
    eval_bs = _env_int('EVAL_BATCH_SIZE', _env_int('INFER_BATCH_SIZE', default_eval_bs))
    min_bs = _env_int('EVAL_MIN_BATCH_SIZE', _FALLBACK_MIN_BATCH_SIZE)
    return max(1, eval_bs), max(1, min_bs)


def resolve_eval_num_workers(train_config: Dict[str, Any]) -> int:
    train_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))
    default_workers = min(max(0, train_workers), _FALLBACK_NUM_WORKERS)
    return max(0, _env_int('EVAL_NUM_WORKERS', default_workers))


def configure_torch_runtime(device: str) -> None:
    torch_threads = _env_int('EVAL_TORCH_NUM_THREADS', 1)
    try:
        torch.set_num_threads(max(1, torch_threads))
    except Exception as exc:
        logging.warning("Failed to set torch num_threads=%s: %s", torch_threads, exc)

    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = _env_flag('EVAL_ALLOW_TF32', True)
            torch.set_float32_matmul_precision(os.environ.get('EVAL_MATMUL_PRECISION', 'high'))
        except Exception as exc:
            logging.warning("Failed to configure CUDA matmul precision: %s", exc)


def worker_init_fn(worker_id: int) -> None:
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    if os.path.isfile(ckpt_path):
        return ckpt_path
    if not os.path.isdir(ckpt_path):
        return None
    preferred = [
        os.path.join(ckpt_path, 'model.pt'),
        os.path.join(ckpt_path, 'best_model.pt'),
        os.path.join(ckpt_path, 'latest_model.pt'),
    ]
    for candidate in preferred:
        if os.path.exists(candidate):
            return candidate
    for item in sorted(os.listdir(ckpt_path)):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert only model-consumed tensors to ``ModelInput``.

    The dataset also returns ``label`` / ``timestamp`` for compatibility with
    training utilities. Moving those unused tensors to GPU during evaluation
    costs time and PCIe bandwidth, so inference transfers only the fields the
    model actually consumes.
    """
    seq_domains = batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_tensor = batch[domain].to(device, non_blocking=True)
        seq_data[domain] = seq_tensor
        seq_lens[domain] = batch[f'{domain}_len'].to(device, non_blocking=True)
        B, _, L = seq_tensor.shape
        time_bucket = batch.get(f'{domain}_time_bucket')
        seq_time_buckets[domain] = (
            time_bucket.to(device, non_blocking=True)
            if isinstance(time_bucket, torch.Tensor) else
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=batch['user_int_feats'].to(device, non_blocking=True),
        item_int_feats=batch['item_int_feats'].to(device, non_blocking=True),
        user_dense_feats=batch['user_dense_feats'].to(device, non_blocking=True),
        item_dense_feats=batch['item_dense_feats'].to(device, non_blocking=True),
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def make_test_dataset(
    data_dir: str,
    schema_path: str,
    batch_size: int,
    seq_max_lens: Dict[str, int],
) -> PCVRParquetDataset:
    return PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )


def make_test_loader(
    test_dataset: PCVRParquetDataset,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(1, prefetch_factor)
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['worker_init_fn'] = worker_init_fn
    return DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


def is_cuda_oom(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return 'cuda' in msg and ('out of memory' in msg or 'cublas' in msg)


def maybe_compile_model(model: nn.Module) -> nn.Module:
    if not _env_flag('EVAL_USE_COMPILE', False):
        return model
    if not hasattr(torch, 'compile'):
        logging.warning("EVAL_USE_COMPILE=1 ignored because torch.compile is unavailable")
        return model
    try:
        logging.info("Compiling model with torch.compile(mode='reduce-overhead')")
        return torch.compile(model, mode='reduce-overhead', fullgraph=False)
    except Exception as exc:
        logging.warning("torch.compile failed; continuing with eager model: %s", exc)
        return model


def run_inference_once(
    model: nn.Module,
    data_dir: str,
    schema_path: str,
    seq_max_lens: Dict[str, int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: str,
) -> Tuple[Dict[Any, float], int]:
    test_dataset = make_test_dataset(data_dir, schema_path, batch_size, seq_max_lens)
    test_loader = make_test_loader(test_dataset, num_workers, prefetch_factor)

    all_probs: List[float] = []
    all_user_ids: List[Any] = []
    raw_count = 0
    start_time = time.time()
    logging.info(
        "Starting inference: batch_size=%s, num_workers=%s, prefetch_factor=%s",
        batch_size,
        num_workers,
        prefetch_factor if num_workers > 0 else 0,
    )

    log_every = max(1, _env_int('EVAL_LOG_EVERY_BATCHES', 100))
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).float().cpu().tolist()
            all_probs.extend(probs)
            all_user_ids.extend(user_ids)
            raw_count += len(probs)

            if (batch_idx + 1) % log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                logging.info(
                    "  Processed %s samples (%.1f samples/s)",
                    raw_count,
                    raw_count / elapsed,
                )

    elapsed = max(time.time() - start_time, 1e-6)
    logging.info(
        "Inference complete: %s rows, %s unique user_ids, %.1f samples/s",
        raw_count,
        len(set(all_user_ids)),
        raw_count / elapsed,
    )
    if len(set(all_user_ids)) != raw_count:
        logging.warning(
            "Duplicate user_id values detected: rows=%s, unique=%s. "
            "predictions.json keeps the last score for each user_id.",
            raw_count,
            len(set(all_user_ids)),
        )
    return dict(zip(all_user_ids, all_probs)), raw_count


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    if not model_dir:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("EVAL_RESULT_PATH is not set")

    model_dir = os.path.dirname(model_dir) if os.path.isfile(model_dir) else model_dir
    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configure_torch_runtime(device)

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: inference batch size is intentionally independent
    # from training batch size. The default is larger, with OOM fallback below.
    batch_size, min_batch_size = resolve_eval_batch_size(train_config)
    num_workers = resolve_eval_num_workers(train_config)
    prefetch_factor = _env_int('EVAL_PREFETCH_FACTOR', 2)
    logging.info(
        "Eval runtime config: batch_size=%s, min_batch_size=%s, num_workers=%s, "
        "prefetch_factor=%s, device=%s",
        batch_size,
        min_batch_size,
        num_workers,
        prefetch_factor if num_workers > 0 else 0,
        device,
    )

    # Build a dataset once for schema / row-count metadata. Retries below may
    # create a new dataset with a smaller batch size.
    metadata_dataset = make_test_dataset(data_dir, schema_path, batch_size, seq_max_lens)
    total_test_samples = metadata_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        metadata_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    model = maybe_compile_model(model)
    model.eval()
    logging.info("Model loaded successfully")

    # Release GPU memory from model construction before inference starts.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_batch_size = batch_size
    while True:
        try:
            predictions_dict, raw_count = run_inference_once(
                model=model,
                data_dir=data_dir,
                schema_path=schema_path,
                seq_max_lens=seq_max_lens,
                batch_size=current_batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                device=device,
            )
            break
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or current_batch_size <= min_batch_size:
                raise
            next_batch_size = max(min_batch_size, current_batch_size // 2)
            logging.warning(
                "CUDA OOM at eval batch_size=%s; retrying from scratch with batch_size=%s",
                current_batch_size,
                next_batch_size,
            )
            current_batch_size = next_batch_size
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump({"predictions": predictions_dict}, f)
    logging.info(
        "Saved %s predictions to %s",
        len(predictions_dict),
        output_path,
    )


if __name__ == "__main__":
    main()
