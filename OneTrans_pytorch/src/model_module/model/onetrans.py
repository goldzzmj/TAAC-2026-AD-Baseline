from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import torch
from torch import nn

from src.model_module.model import register_model
from src.model_module.model.layers import RMSNorm, OneTransBlock, build_pyramid_schedule
from src.model_module.model.schema import ModelConfig
from src.model_module.model.tokenizer import NonSequentialTokenizer, SequentialTokenizer


class OneTransBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_ns_tokens = cfg.resolved_ns_token_count
        self.non_sequential_tokenizer = NonSequentialTokenizer(cfg)
        self.sequential_tokenizer = SequentialTokenizer(cfg)
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    ffn_hidden_dim=cfg.ffn_hidden_dim,
                    num_ns_tokens=self.num_ns_tokens,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.output_norm = RMSNorm(cfg.hidden_dim)

    def _validate_inputs(self, batch: Dict[str, Any]) -> None:
        if "non_sequential" not in batch or "sequential" not in batch:
            raise KeyError("Batch must contain 'non_sequential' and 'sequential'.")
        for feature_name in self.cfg.non_sequential_feature_names:
            if feature_name not in batch["non_sequential"]:
                raise KeyError(f"Missing non-sequential feature: {feature_name}")
        for sequence_name in self.cfg.sequential_feature_names:
            if sequence_name not in batch["sequential"]:
                raise KeyError(f"Missing sequence feature: {sequence_name}")
            sequence_inputs = batch["sequential"][sequence_name]
            if "values" not in sequence_inputs or "mask" not in sequence_inputs:
                raise KeyError(f"Sequence '{sequence_name}' requires 'values' and 'mask'.")
            if self.cfg.sequence_merge_type == "timestamp_aware" and "timestamps" not in sequence_inputs:
                raise KeyError(f"Sequence '{sequence_name}' requires 'timestamps' in timestamp_aware mode.")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self._validate_inputs(batch)
        ns_tokens = self.non_sequential_tokenizer(batch["non_sequential"])
        seq_tokens, seq_mask = self.sequential_tokenizer(batch["sequential"])

        batch_size = ns_tokens.size(0)
        ns_mask = torch.ones((batch_size, self.num_ns_tokens), device=ns_tokens.device, dtype=torch.bool)
        hidden = torch.cat([seq_tokens, ns_tokens], dim=1)
        token_mask = torch.cat([seq_mask, ns_mask], dim=1)
        num_seq_tokens = seq_tokens.size(1)

        schedule = build_pyramid_schedule(
            initial_seq_tokens=num_seq_tokens,
            num_ns_tokens=self.num_ns_tokens,
            num_layers=self.cfg.num_layers,
            final_seq_query_tokens=self.cfg.resolved_final_seq_query_tokens,
            round_to=self.cfg.pyramid_round_to,
            use_pyramid=self.cfg.use_pyramid,
        )

        for block, query_keep_total in zip(self.blocks, schedule):
            hidden, token_mask, num_seq_tokens = block(
                inputs=hidden,
                token_mask=token_mask,
                num_seq_tokens=num_seq_tokens,
                query_keep_total=query_keep_total,
            )

        hidden = self.output_norm(hidden)
        non_sequential_tokens = hidden[:, -self.num_ns_tokens :]
        sequence_tokens = hidden[:, :num_seq_tokens]
        return {
            "tokens": hidden,
            "token_mask": token_mask,
            "sequence_tokens": sequence_tokens,
            "non_sequential_tokens": non_sequential_tokens,
            "sequence_token_count": torch.tensor(num_seq_tokens, device=hidden.device),
        }


@register_model("onetrans")
class OneTrans(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = OneTransBackbone(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.resolved_ns_token_count * cfg.hidden_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.output_dim),
        )

    def forward(
        self,
        batch: Dict[str, Any],
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(batch)
        non_sequential_tokens = backbone_outputs["non_sequential_tokens"]
        flattened = non_sequential_tokens.flatten(start_dim=1)
        logits = self.head(flattened)

        outputs: Dict[str, torch.Tensor] = {
            **backbone_outputs,
            "logits": logits,
        }
        if labels is not None:
            if self.cfg.output_dim == 1:
                target = labels.float().view(-1, 1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            else:
                loss = nn.functional.cross_entropy(logits, labels.long())
            outputs["loss"] = loss
            outputs["labels"] = labels
        return outputs

    def export_config(self) -> Dict[str, Any]:
        return asdict(self.cfg)
