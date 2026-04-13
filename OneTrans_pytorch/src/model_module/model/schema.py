from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TokenizerType = Literal["auto_split", "groupwise"]
SequenceMergeType = Literal["timestamp_aware", "timestamp_agnostic"]


@dataclass(frozen=True)
class NonSequentialFeatureSpec:
    name: str
    input_dim: int


@dataclass(frozen=True)
class SequentialFeatureSpec:
    name: str
    input_dim: int
    impact_order: int = 0


@dataclass(frozen=True)
class ModelConfig:
    non_sequential_features: tuple[NonSequentialFeatureSpec, ...]
    sequential_features: tuple[SequentialFeatureSpec, ...]
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ffn_hidden_dim: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    ns_tokenizer_type: TokenizerType = "auto_split"
    ns_token_count: int = 12
    groupwise_feature_groups: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    sequence_merge_type: SequenceMergeType = "timestamp_aware"
    use_sep_token: bool = True
    use_sequence_type_embedding: bool = True
    use_pyramid: bool = True
    final_seq_query_tokens: int | None = None
    pyramid_round_to: int = 32
    output_dim: int = 1
    head_hidden_dim: int = 256

    def __post_init__(self) -> None:
        ns_names = {spec.name for spec in self.non_sequential_features}
        if not self.non_sequential_features:
            raise ValueError("At least one non-sequential feature is required.")
        if not self.sequential_features:
            raise ValueError("At least one sequential feature is required.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive.")
        if self.ns_tokenizer_type == "auto_split" and self.ns_token_count <= 0:
            raise ValueError("ns_token_count must be positive for auto_split.")
        if self.ns_tokenizer_type == "groupwise" and not self.groupwise_feature_groups:
            raise ValueError("groupwise_feature_groups is required for groupwise tokenizer.")
        if self.ns_tokenizer_type == "groupwise":
            unknown_names = set().union(*self.groupwise_feature_groups) - ns_names
            if unknown_names:
                raise ValueError(f"Unknown groupwise feature names: {sorted(unknown_names)}")
        if self.pyramid_round_to <= 0:
            raise ValueError("pyramid_round_to must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

    @property
    def non_sequential_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.non_sequential_features)

    @property
    def sequential_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.sequential_features)

    @property
    def resolved_ns_token_count(self) -> int:
        if self.ns_tokenizer_type == "groupwise":
            return len(self.groupwise_feature_groups)
        return self.ns_token_count

    @property
    def resolved_final_seq_query_tokens(self) -> int:
        if self.final_seq_query_tokens is not None:
            return self.final_seq_query_tokens
        return self.resolved_ns_token_count
