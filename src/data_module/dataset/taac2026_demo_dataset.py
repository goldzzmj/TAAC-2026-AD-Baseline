from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_module.dataset import register_dataset


DOMAIN_PREFIXES = (
    "domain_a_seq_",
    "domain_b_seq_",
    "domain_c_seq_",
    "domain_d_seq_",
)
LABEL_COLUMN = "label_type"
EXCLUDED_COLUMNS = {"label_type", "label_time"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value):
        return True
    return False


def _first_non_missing(values: Iterable[Any]) -> Any:
    for value in values:
        if _is_missing(value):
            continue
        return value
    return None


def _to_int(value: Any) -> int:
    if _is_missing(value):
        return 0
    return int(value)


def _to_int_list(value: Any) -> list[int]:
    if _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _to_float_list(value: Any, expected_dim: int) -> list[float]:
    if _is_missing(value):
        return [0.0] * expected_dim
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [float(value)]
    if len(values) < expected_dim:
        values.extend([0.0] * (expected_dim - len(values)))
    return [float(v) for v in values[:expected_dim]]


def _infer_vector_length(value: Any) -> int:
    if _is_missing(value):
        return 0
    if isinstance(value, np.ndarray):
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


@dataclass
class FeatureSchema:
    label_column: str
    label_values: tuple[int, ...]
    positive_label_value: int
    scalar_fields: tuple[str, ...]
    list_int_fields: tuple[str, ...]
    dense_fields: tuple[str, ...]
    domain_fields: dict[str, tuple[str, ...]]
    dense_dims: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_column": self.label_column,
            "label_values": list(self.label_values),
            "positive_label_value": self.positive_label_value,
            "scalar_fields": list(self.scalar_fields),
            "list_int_fields": list(self.list_int_fields),
            "dense_fields": list(self.dense_fields),
            "domain_fields": {k: list(v) for k, v in self.domain_fields.items()},
            "dense_dims": dict(self.dense_dims),
        }


def build_feature_schema(
    frame: pd.DataFrame,
    positive_label_value: int | None = None,
) -> FeatureSchema:
    label_values = tuple(sorted(int(v) for v in frame[LABEL_COLUMN].dropna().unique().tolist()))
    if not label_values:
        raise ValueError("No label values were found in the parquet file.")
    positive = positive_label_value if positive_label_value is not None else label_values[-1]

    dense_fields = tuple(sorted(col for col in frame.columns if col.startswith("user_dense_feats_")))
    domain_fields: dict[str, tuple[str, ...]] = {}
    domain_columns: set[str] = set()
    for prefix in DOMAIN_PREFIXES:
        matched = tuple(sorted(col for col in frame.columns if col.startswith(prefix)))
        if matched:
            domain_name = prefix.replace("_seq_", "")
            domain_fields[domain_name] = matched
            domain_columns.update(matched)

    scalar_fields: list[str] = []
    list_int_fields: list[str] = []
    dense_dims: dict[str, int] = {}

    for field in frame.columns:
        if field in EXCLUDED_COLUMNS or field in domain_columns:
            continue
        if field in dense_fields:
            sample = _first_non_missing(frame[field].tolist())
            dense_dims[field] = _infer_vector_length(sample)
            continue

        sample = _first_non_missing(frame[field].tolist())
        if isinstance(sample, np.ndarray):
            sample = sample.tolist()

        if isinstance(sample, (list, tuple)):
            list_int_fields.append(field)
        else:
            scalar_fields.append(field)

    for field in dense_fields:
        if field not in dense_dims:
            sample = _first_non_missing(frame[field].tolist())
            dense_dims[field] = _infer_vector_length(sample)

    return FeatureSchema(
        label_column=LABEL_COLUMN,
        label_values=label_values,
        positive_label_value=int(positive),
        scalar_fields=tuple(sorted(scalar_fields)),
        list_int_fields=tuple(sorted(list_int_fields)),
        dense_fields=tuple(sorted(dense_fields)),
        domain_fields=domain_fields,
        dense_dims=dense_dims,
    )


@register_dataset("taac2026_demo")
class TAAC2026DemoDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        schema: FeatureSchema | None = None,
        positive_label_value: int | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.frame = pd.read_parquet(self.parquet_path)
        self.schema = schema or build_feature_schema(self.frame, positive_label_value=positive_label_value)
        self.label_to_index = {label: index for index, label in enumerate(self.schema.label_values)}
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}
        self.positive_label_index = self.label_to_index[self.schema.positive_label_value]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        raw_label = int(row[self.schema.label_column])
        sample = {
            "label": self.label_to_index[raw_label],
            "raw_label": raw_label,
            "binary_label": 1 if raw_label == self.schema.positive_label_value else 0,
            "scalar_features": {},
            "list_int_features": {},
            "dense_features": {},
            "domain_features": {},
        }

        for field in self.schema.scalar_fields:
            sample["scalar_features"][field] = _to_int(row[field])

        for field in self.schema.list_int_fields:
            sample["list_int_features"][field] = _to_int_list(row[field])

        for field in self.schema.dense_fields:
            expected_dim = self.schema.dense_dims[field]
            sample["dense_features"][field] = _to_float_list(row[field], expected_dim=expected_dim)

        for domain, fields in self.schema.domain_fields.items():
            sample["domain_features"][domain] = {field: _to_int_list(row[field]) for field in fields}

        return sample


def _pad_int_lists(values: list[list[int]], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros((len(values), max_length), dtype=torch.long)
    mask = torch.zeros((len(values), max_length), dtype=torch.bool)
    for row_index, row_values in enumerate(values):
        clipped = row_values[:max_length]
        if clipped:
            padded[row_index, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            mask[row_index, : len(clipped)] = True
    return padded, mask


def _stack_dense(values: list[list[float]], dim: int) -> torch.Tensor:
    return torch.tensor([row[:dim] for row in values], dtype=torch.float32)


def create_taac2026_collate_fn(
    schema: FeatureSchema,
    max_list_length: int = 8,
    max_seq_length: int = 128,
):
    def collate_fn(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "binary_labels": torch.tensor([sample["binary_label"] for sample in samples], dtype=torch.float32),
            "raw_labels": torch.tensor([sample["raw_label"] for sample in samples], dtype=torch.long),
            "scalar_features": {},
            "list_int_features": {},
            "list_int_masks": {},
            "dense_features": {},
            "domain_features": {},
            "domain_masks": {},
        }

        for field in schema.scalar_fields:
            values = [sample["scalar_features"][field] for sample in samples]
            batch["scalar_features"][field] = torch.tensor(values, dtype=torch.long)

        for field in schema.list_int_fields:
            values = [sample["list_int_features"][field] for sample in samples]
            padded, mask = _pad_int_lists(values, max_length=max_list_length)
            batch["list_int_features"][field] = padded
            batch["list_int_masks"][field] = mask

        for field in schema.dense_fields:
            values = [sample["dense_features"][field] for sample in samples]
            batch["dense_features"][field] = _stack_dense(values, dim=schema.dense_dims[field])

        for domain, fields in schema.domain_fields.items():
            batch["domain_features"][domain] = {}
            domain_lengths: list[int] = []
            for sample in samples:
                lengths = [len(sample["domain_features"][domain][field]) for field in fields]
                domain_lengths.append(max(lengths) if lengths else 0)

            domain_mask = torch.zeros((len(samples), max_seq_length), dtype=torch.bool)
            for row_index, length in enumerate(domain_lengths):
                domain_mask[row_index, : min(length, max_seq_length)] = True
            batch["domain_masks"][domain] = domain_mask

            for field in fields:
                values = [sample["domain_features"][domain][field] for sample in samples]
                padded, _ = _pad_int_lists(values, max_length=max_seq_length)
                batch["domain_features"][domain][field] = padded

        return batch

    return collate_fn
