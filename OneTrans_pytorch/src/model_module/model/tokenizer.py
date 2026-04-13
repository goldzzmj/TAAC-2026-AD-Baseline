from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.model_module.model.schema import ModelConfig, SequentialFeatureSpec


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class AutoSplitNonSequentialTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_names = cfg.non_sequential_feature_names
        total_input_dim = sum(spec.input_dim for spec in cfg.non_sequential_features)
        self.projection = ProjectionMLP(
            input_dim=total_input_dim,
            output_dim=cfg.resolved_ns_token_count * cfg.hidden_dim,
            dropout=cfg.dropout,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat([features[name].float() for name in self.feature_names], dim=-1)
        tokens = self.projection(concatenated)
        batch_size = concatenated.size(0)
        return tokens.view(batch_size, self.cfg.resolved_ns_token_count, self.cfg.hidden_dim)


class GroupWiseNonSequentialTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        feature_dim_map = {spec.name: spec.input_dim for spec in cfg.non_sequential_features}
        self.groups = cfg.groupwise_feature_groups
        self.group_mlps = nn.ModuleList()
        for group in self.groups:
            group_dim = sum(feature_dim_map[name] for name in group)
            self.group_mlps.append(ProjectionMLP(group_dim, cfg.hidden_dim, cfg.dropout))

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = []
        for group, mlp in zip(self.groups, self.group_mlps):
            group_tensor = torch.cat([features[name].float() for name in group], dim=-1)
            outputs.append(mlp(group_tensor).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class NonSequentialTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.ns_tokenizer_type == "groupwise":
            self.impl = GroupWiseNonSequentialTokenizer(cfg)
        else:
            self.impl = AutoSplitNonSequentialTokenizer(cfg)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.impl(features)


class SequentialTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.sequence_specs = cfg.sequential_features
        self.sequence_projectors = nn.ModuleDict(
            {
                spec.name: ProjectionMLP(spec.input_dim, cfg.hidden_dim, cfg.dropout)
                for spec in self.sequence_specs
            }
        )
        if cfg.use_sequence_type_embedding:
            self.sequence_type_embeddings = nn.Embedding(len(self.sequence_specs), cfg.hidden_dim)
        else:
            self.sequence_type_embeddings = None
        self.sep_token = nn.Parameter(torch.zeros(cfg.hidden_dim))
        nn.init.normal_(self.sep_token, mean=0.0, std=0.02)

    def _project_sequences(self, sequences: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        for spec in self.sequence_specs:
            values = sequences[spec.name]["values"].float()
            projected[spec.name] = self.sequence_projectors[spec.name](values)
        return projected

    def _add_type_embedding(self, tokens: torch.Tensor, sequence_index: int) -> torch.Tensor:
        if self.sequence_type_embeddings is None:
            return tokens
        type_embedding = self.sequence_type_embeddings.weight[sequence_index]
        return tokens + type_embedding

    def _merge_timestamp_aware(
        self,
        projected: Dict[str, torch.Tensor],
        sequences: Dict[str, Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = next(iter(projected.values())).size(0)
        device = next(iter(projected.values())).device
        dtype = next(iter(projected.values())).dtype
        merged_samples = []
        max_length = 0

        for batch_index in range(batch_size):
            events = []
            for sequence_index, spec in enumerate(self.sequence_specs):
                seq_inputs = sequences[spec.name]
                seq_mask = seq_inputs["mask"][batch_index].bool()
                seq_timestamps = seq_inputs["timestamps"][batch_index]
                seq_tokens = projected[spec.name][batch_index]
                valid_indices = torch.nonzero(seq_mask, as_tuple=False).flatten().tolist()
                for item_index in valid_indices:
                    token = seq_tokens[item_index]
                    token = self._add_type_embedding(token, sequence_index)
                    events.append((float(seq_timestamps[item_index].item()), item_index, token))

            events.sort(key=lambda item: (item[0], item[1]))
            if events:
                sample_tokens = torch.stack([item[2] for item in events], dim=0)
            else:
                sample_tokens = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            merged_samples.append(sample_tokens)
            max_length = max(max_length, sample_tokens.size(0))

        if max_length == 0:
            empty_tokens = torch.zeros((batch_size, 0, self.hidden_dim), device=device, dtype=dtype)
            empty_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            return empty_tokens, empty_mask

        padded = torch.zeros((batch_size, max_length, self.hidden_dim), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_length), device=device, dtype=torch.bool)
        for batch_index, sample_tokens in enumerate(merged_samples):
            sample_length = sample_tokens.size(0)
            if sample_length > 0:
                padded[batch_index, :sample_length] = sample_tokens
                mask[batch_index, :sample_length] = True
        return padded, mask

    def _ordered_specs(self) -> list[SequentialFeatureSpec]:
        return sorted(self.sequence_specs, key=lambda spec: spec.impact_order, reverse=True)

    def _merge_timestamp_agnostic(
        self,
        projected: Dict[str, torch.Tensor],
        sequences: Dict[str, Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = next(iter(projected.values())).size(0)
        device = next(iter(projected.values())).device
        dtype = next(iter(projected.values())).dtype
        ordered_specs = self._ordered_specs()
        merged_samples = []
        max_length = 0

        for batch_index in range(batch_size):
            chunks = []
            for spec in ordered_specs:
                seq_mask = sequences[spec.name]["mask"][batch_index].bool()
                seq_tokens = projected[spec.name][batch_index][seq_mask]
                if seq_tokens.numel() == 0:
                    continue
                chunks.append(seq_tokens)

            if not chunks:
                sample_tokens = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            else:
                sample_parts = []
                for part_index, seq_tokens in enumerate(chunks):
                    sample_parts.append(seq_tokens)
                    is_last = part_index == len(chunks) - 1
                    if self.cfg.use_sep_token and not is_last:
                        sample_parts.append(self.sep_token.view(1, -1).to(device=device, dtype=dtype))
                sample_tokens = torch.cat(sample_parts, dim=0)

            merged_samples.append(sample_tokens)
            max_length = max(max_length, sample_tokens.size(0))

        if max_length == 0:
            empty_tokens = torch.zeros((batch_size, 0, self.hidden_dim), device=device, dtype=dtype)
            empty_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            return empty_tokens, empty_mask

        padded = torch.zeros((batch_size, max_length, self.hidden_dim), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_length), device=device, dtype=torch.bool)
        for batch_index, sample_tokens in enumerate(merged_samples):
            sample_length = sample_tokens.size(0)
            if sample_length > 0:
                padded[batch_index, :sample_length] = sample_tokens
                mask[batch_index, :sample_length] = True
        return padded, mask

    def forward(self, sequences: Dict[str, Dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self._project_sequences(sequences)
        if self.cfg.sequence_merge_type == "timestamp_agnostic":
            return self._merge_timestamp_agnostic(projected, sequences)
        return self._merge_timestamp_aware(projected, sequences)
