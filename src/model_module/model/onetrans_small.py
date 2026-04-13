from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data_module.dataset.taac2026_demo_dataset import FeatureSchema
from src.model_module.model import register_model


@dataclass(frozen=True)
class OneTransSmallConfig:
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 192
    dropout: float = 0.1
    hash_bucket_size: int = 50000
    max_seq_length: int = 128


@register_model("onetrans_small")
class OneTransSmall(nn.Module):
    def __init__(self, schema: FeatureSchema, cfg: OneTransSmallConfig) -> None:
        super().__init__()
        self.schema = schema
        self.cfg = cfg
        self.hash_bucket_size = cfg.hash_bucket_size
        self.non_seq_fields = list(schema.scalar_fields + schema.list_int_fields + schema.dense_fields)
        self.domain_names = list(schema.domain_fields.keys())

        self.scalar_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(cfg.hash_bucket_size + 1, cfg.hidden_dim, padding_idx=0)
                for field in schema.scalar_fields
            }
        )
        self.list_int_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(cfg.hash_bucket_size + 1, cfg.hidden_dim, padding_idx=0)
                for field in schema.list_int_fields
            }
        )
        self.domain_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(cfg.hash_bucket_size + 1, cfg.hidden_dim, padding_idx=0)
                for fields in schema.domain_fields.values()
                for field in fields
            }
        )
        self.dense_projections = nn.ModuleDict(
            {field: nn.Linear(schema.dense_dims[field], cfg.hidden_dim) for field in schema.dense_fields}
        )

        self.field_index = {field: index + 1 for index, field in enumerate(self.non_seq_fields)}
        self.domain_index = {domain: index + 1 for index, domain in enumerate(self.domain_names)}

        self.field_embeddings = nn.Embedding(len(self.field_index) + 1, cfg.hidden_dim)
        self.domain_type_embeddings = nn.Embedding(len(self.domain_index) + 1, cfg.hidden_dim)
        self.position_embeddings = nn.Embedding(cfg.max_seq_length + 1, cfg.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=cfg.num_layers)
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, len(schema.label_values))

    def _hash_ids(self, values: torch.Tensor) -> torch.Tensor:
        values = values.long()
        zeros = torch.zeros_like(values)
        positive = torch.clamp(values, min=0)
        hashed = torch.remainder(positive, self.hash_bucket_size - 1) + 1
        return torch.where(values > 0, hashed, zeros)

    def _pooled_list_token(self, field: str, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hashed = self._hash_ids(values)
        emb = self.list_int_embeddings[field](hashed)
        weight = mask.unsqueeze(-1).float()
        pooled = (emb * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        return pooled

    def _field_token(self, field: str, token: torch.Tensor, device: torch.device) -> torch.Tensor:
        field_index = torch.full(
            (token.size(0),),
            fill_value=self.field_index[field],
            dtype=torch.long,
            device=device,
        )
        return token + self.field_embeddings(field_index)

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        sample_field = next(iter(batch["scalar_features"].values()))
        device = sample_field.device
        batch_size = sample_field.size(0)

        token_blocks: list[torch.Tensor] = []
        mask_blocks: list[torch.Tensor] = []

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        token_blocks.append(cls_token)
        mask_blocks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))

        for field in self.schema.scalar_fields:
            values = batch["scalar_features"][field].to(device)
            token = self.scalar_embeddings[field](self._hash_ids(values))
            token = self._field_token(field, token, device)
            token_blocks.append(token.unsqueeze(1))
            mask_blocks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))

        for field in self.schema.list_int_fields:
            values = batch["list_int_features"][field].to(device)
            mask = batch["list_int_masks"][field].to(device)
            token = self._pooled_list_token(field, values, mask)
            token = self._field_token(field, token, device)
            token_blocks.append(token.unsqueeze(1))
            mask_blocks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))

        for field in self.schema.dense_fields:
            values = batch["dense_features"][field].to(device)
            token = self.dense_projections[field](values)
            token = self._field_token(field, token, device)
            token_blocks.append(token.unsqueeze(1))
            mask_blocks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))

        position_index = torch.arange(self.cfg.max_seq_length, device=device).unsqueeze(0)
        position_emb = self.position_embeddings(position_index)

        for domain, fields in self.schema.domain_fields.items():
            seq_mask = batch["domain_masks"][domain].to(device)
            field_tokens = []
            for field in fields:
                values = batch["domain_features"][domain][field].to(device)
                token = self.domain_embeddings[field](self._hash_ids(values))
                field_tokens.append(token)
            domain_token = torch.stack(field_tokens, dim=0).sum(dim=0)
            domain_index = torch.full(
                (batch_size, self.cfg.max_seq_length),
                fill_value=self.domain_index[domain],
                dtype=torch.long,
                device=device,
            )
            domain_token = domain_token + self.domain_type_embeddings(domain_index) + position_emb
            token_blocks.append(domain_token)
            mask_blocks.append(seq_mask)

        tokens = torch.cat(token_blocks, dim=1)
        token_mask = torch.cat(mask_blocks, dim=1)
        tokens = self.input_norm(self.dropout(tokens))
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        cls_state = self.output_norm(encoded[:, 0])
        logits = self.head(self.dropout(cls_state))
        return {"logits": logits}
