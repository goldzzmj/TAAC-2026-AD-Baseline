"""Helpers for building a PCVR schema.json directly from parquet files."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
import pyarrow.types as pat


_USER_INT_PREFIX = "user_int_feats_"
_USER_DENSE_PREFIX = "user_dense_feats_"
_ITEM_INT_PREFIX = "item_int_feats_"
_DOMAIN_PREFIX_RE = re.compile(r"^(domain_[a-z]+_seq)_(\d+)$")
_FID_RE = re.compile(r"(\d+)$")
_TIMESTAMP_FLOOR = 1_500_000_000


@dataclass
class ColumnStats:
    name: str
    max_len: int
    max_positive: int
    min_non_null: Optional[float]
    max_non_null: Optional[float]
    is_list: bool
    is_floating_list: bool


def _iter_parquet_files(data_path: str) -> List[str]:
    if os.path.isdir(data_path):
        files = sorted(
            os.path.join(data_path, name)
            for name in os.listdir(data_path)
            if name.endswith(".parquet")
        )
        if not files:
            raise FileNotFoundError(f"No .parquet files found under {data_path}")
        return files
    if os.path.isfile(data_path):
        return [data_path]
    raise FileNotFoundError(f"Parquet path does not exist: {data_path}")


def _extract_fid(column_name: str) -> int:
    match = _FID_RE.search(column_name)
    if not match:
        raise ValueError(f"Failed to parse feature id from column name: {column_name}")
    return int(match.group(1))


def _safe_int_vocab(max_positive: int) -> int:
    if max_positive <= 0:
        return 0
    return max_positive + 1


def _scan_column_stats(data_path: str) -> Dict[str, ColumnStats]:
    stats: Dict[str, ColumnStats] = {}
    for parquet_file in _iter_parquet_files(data_path):
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches():
            for idx, name in enumerate(batch.schema.names):
                column = batch.column(idx)
                col_type = column.type

                entry = stats.get(name)
                if entry is None:
                    entry = ColumnStats(
                        name=name,
                        max_len=0,
                        max_positive=0,
                        min_non_null=None,
                        max_non_null=None,
                        is_list=pat.is_list(col_type),
                        is_floating_list=pat.is_list(col_type) and pat.is_floating(col_type.value_type),
                    )
                    stats[name] = entry

                if pat.is_list(col_type):
                    chunks = column.chunks if hasattr(column, "chunks") else [column]
                    for chunk in chunks:
                        if len(chunk) == 0:
                            continue
                        offsets = chunk.offsets.to_numpy()
                        if len(offsets) > 1:
                            entry.max_len = max(
                                entry.max_len,
                                int((offsets[1:] - offsets[:-1]).max()),
                            )
                        values = chunk.values.to_numpy(zero_copy_only=False)
                        if len(values) == 0:
                            continue
                        chunk_min = float(values.min())
                        chunk_max = float(values.max())
                        if entry.min_non_null is None or chunk_min < entry.min_non_null:
                            entry.min_non_null = chunk_min
                        if entry.max_non_null is None or chunk_max > entry.max_non_null:
                            entry.max_non_null = chunk_max
                        if not entry.is_floating_list:
                            positive_values = values[values > 0]
                            if len(positive_values) > 0:
                                entry.max_positive = max(
                                    entry.max_positive,
                                    int(positive_values.max()),
                                )
                else:
                    array = column.to_numpy(zero_copy_only=False)
                    valid_values = []
                    for value in array.tolist():
                        if value is None:
                            continue
                        if isinstance(value, float) and math.isnan(value):
                            continue
                        valid_values.append(value)
                    if not valid_values:
                        continue
                    chunk_min = float(min(valid_values))
                    chunk_max = float(max(valid_values))
                    if entry.min_non_null is None or chunk_min < entry.min_non_null:
                        entry.min_non_null = chunk_min
                    if entry.max_non_null is None or chunk_max > entry.max_non_null:
                        entry.max_non_null = chunk_max
                    positive_values = [int(v) for v in valid_values if float(v) > 0]
                    if positive_values:
                        entry.max_positive = max(entry.max_positive, max(positive_values))
    return stats


def _build_int_feature_list(
    stats: Dict[str, ColumnStats],
    prefix: str,
) -> List[List[int]]:
    features: List[List[int]] = []
    for name in sorted(col for col in stats if col.startswith(prefix)):
        col_stats = stats[name]
        fid = _extract_fid(name)
        dim = max(1, col_stats.max_len) if col_stats.is_list else 1
        vocab_size = _safe_int_vocab(col_stats.max_positive)
        features.append([fid, vocab_size, dim])
    return features


def _build_dense_feature_list(stats: Dict[str, ColumnStats]) -> List[List[int]]:
    features: List[List[int]] = []
    for name in sorted(col for col in stats if col.startswith(_USER_DENSE_PREFIX)):
        col_stats = stats[name]
        fid = _extract_fid(name)
        dim = max(1, col_stats.max_len)
        features.append([fid, dim])
    return features


def _guess_ts_fid(domain_columns: Iterable[Tuple[int, ColumnStats]]) -> Optional[int]:
    candidates = [
        fid
        for fid, col_stats in domain_columns
        if col_stats.max_non_null is not None
        and col_stats.max_non_null >= _TIMESTAMP_FLOOR
        and col_stats.min_non_null is not None
        and col_stats.min_non_null >= _TIMESTAMP_FLOOR
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return min(candidates)
    return None


def _build_seq_feature_dict(stats: Dict[str, ColumnStats]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[int, ColumnStats]]] = {}
    for name, col_stats in stats.items():
        match = _DOMAIN_PREFIX_RE.match(name)
        if not match:
            continue
        prefix = match.group(1)
        fid = int(match.group(2))
        grouped.setdefault(prefix, []).append((fid, col_stats))

    seq: Dict[str, Dict[str, Any]] = {}
    for prefix in sorted(grouped):
        columns = sorted(grouped[prefix], key=lambda item: item[0])
        ts_fid = _guess_ts_fid(columns)
        features = []
        for fid, col_stats in columns:
            vocab_size = 0 if fid == ts_fid else _safe_int_vocab(col_stats.max_positive)
            features.append([fid, vocab_size])
        domain_token = prefix.replace("domain_", "").replace("_seq", "")
        domain_name = f"seq_{domain_token}"
        seq[domain_name] = {
            "prefix": prefix,
            "ts_fid": ts_fid,
            "features": features,
        }
        logging.info(
            "Schema domain %s: %d features, ts_fid=%s",
            domain_name,
            len(features),
            ts_fid,
        )
    return seq


def build_schema_dict_from_parquet(data_path: str) -> Dict[str, Any]:
    stats = _scan_column_stats(data_path)
    schema = {
        "user_int": _build_int_feature_list(stats, _USER_INT_PREFIX),
        "item_int": _build_int_feature_list(stats, _ITEM_INT_PREFIX),
        "user_dense": _build_dense_feature_list(stats),
        "seq": _build_seq_feature_dict(stats),
    }
    return schema


def ensure_schema_file(data_path: str, schema_path: str, overwrite: bool = False) -> str:
    if os.path.exists(schema_path) and not overwrite:
        logging.info("Using existing schema file: %s", schema_path)
        return schema_path

    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    schema = build_schema_dict_from_parquet(data_path)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    logging.info("Generated schema file: %s", schema_path)
    return schema_path
