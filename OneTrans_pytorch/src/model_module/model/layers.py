from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MixedCausalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_ns_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_ns_tokens = num_ns_tokens
        self.head_dim = hidden_dim // num_heads

        self.q_proj_seq = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_seq = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_seq = nn.Linear(hidden_dim, hidden_dim)

        self.q_proj_ns = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_ns_tokens))
        self.k_proj_ns = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_ns_tokens))
        self.v_proj_ns = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_ns_tokens))

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _project_mixed(self, inputs: torch.Tensor, num_seq_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.size(0)
        seq_part = inputs[:, :num_seq_tokens]
        ns_part = inputs[:, num_seq_tokens:]

        if num_seq_tokens > 0:
            q_seq = self.q_proj_seq(seq_part)
            k_seq = self.k_proj_seq(seq_part)
            v_seq = self.v_proj_seq(seq_part)
        else:
            q_seq = inputs.new_zeros((batch_size, 0, self.hidden_dim))
            k_seq = inputs.new_zeros((batch_size, 0, self.hidden_dim))
            v_seq = inputs.new_zeros((batch_size, 0, self.hidden_dim))

        q_ns_list = []
        k_ns_list = []
        v_ns_list = []
        for token_index in range(self.num_ns_tokens):
            ns_token = ns_part[:, token_index : token_index + 1]
            q_ns_list.append(self.q_proj_ns[token_index](ns_token))
            k_ns_list.append(self.k_proj_ns[token_index](ns_token))
            v_ns_list.append(self.v_proj_ns[token_index](ns_token))

        q_ns = torch.cat(q_ns_list, dim=1)
        k_ns = torch.cat(k_ns_list, dim=1)
        v_ns = torch.cat(v_ns_list, dim=1)
        return torch.cat([q_seq, q_ns], dim=1), torch.cat([k_seq, k_ns], dim=1), torch.cat([v_seq, v_ns], dim=1)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        token_mask: torch.Tensor,
        num_seq_tokens: int,
        query_keep_total: int,
    ) -> torch.Tensor:
        total_tokens = inputs.size(1)
        query_start = total_tokens - query_keep_total
        query_mask = token_mask[:, query_start:]

        q_full, k_full, v_full = self._project_mixed(inputs, num_seq_tokens)
        q = self._reshape_heads(q_full[:, query_start:])
        k = self._reshape_heads(k_full)
        v = self._reshape_heads(v_full)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_positions = torch.arange(total_tokens, device=inputs.device)
        query_positions = torch.arange(query_start, total_tokens, device=inputs.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        valid_matrix = causal_mask.unsqueeze(0).unsqueeze(0)
        valid_matrix = valid_matrix & token_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~valid_matrix, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights * query_mask.unsqueeze(1).unsqueeze(-1).float()
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(inputs.size(0), query_keep_total, self.hidden_dim)
        output = self.out_proj(context)
        return output * query_mask.unsqueeze(-1).float()


class MixedFFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_hidden_dim: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.num_ns_tokens = num_ns_tokens
        self.shared_ffn = FeedForward(hidden_dim, ffn_hidden_dim, dropout)
        self.token_specific_ffn = nn.ModuleList(
            FeedForward(hidden_dim, ffn_hidden_dim, dropout) for _ in range(num_ns_tokens)
        )

    def forward(self, inputs: torch.Tensor, num_seq_tokens: int) -> torch.Tensor:
        batch_size, total_tokens, hidden_dim = inputs.shape
        seq_part = inputs[:, :num_seq_tokens]
        ns_part = inputs[:, num_seq_tokens:]

        if num_seq_tokens > 0:
            seq_output = self.shared_ffn(seq_part)
        else:
            seq_output = inputs.new_zeros((batch_size, 0, hidden_dim))

        ns_outputs = []
        for token_index in range(self.num_ns_tokens):
            ns_token = ns_part[:, token_index : token_index + 1]
            ns_outputs.append(self.token_specific_ffn[token_index](ns_token))
        ns_output = torch.cat(ns_outputs, dim=1)
        return torch.cat([seq_output, ns_output], dim=1)


class OneTransBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_ns_tokens: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.num_ns_tokens = num_ns_tokens
        self.attn_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.attn = MixedCausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_ns_tokens=num_ns_tokens,
            dropout=attention_dropout,
        )
        self.ffn = MixedFFN(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            num_ns_tokens=num_ns_tokens,
            dropout=dropout,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        token_mask: torch.Tensor,
        num_seq_tokens: int,
        query_keep_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if query_keep_total < self.num_ns_tokens or query_keep_total > inputs.size(1):
            raise ValueError("query_keep_total must be within the current token range.")

        tail_inputs = inputs[:, -query_keep_total:]
        tail_mask = token_mask[:, -query_keep_total:]
        attn_output = self.attn(
            inputs=self.attn_norm(inputs),
            token_mask=token_mask,
            num_seq_tokens=num_seq_tokens,
            query_keep_total=query_keep_total,
        )
        hidden = tail_inputs + self.residual_dropout(attn_output)
        hidden = hidden * tail_mask.unsqueeze(-1).float()

        next_seq_tokens = max(query_keep_total - self.num_ns_tokens, 0)
        ffn_output = self.ffn(self.ffn_norm(hidden), num_seq_tokens=next_seq_tokens)
        hidden = hidden + self.residual_dropout(ffn_output)
        hidden = hidden * tail_mask.unsqueeze(-1).float()
        return hidden, tail_mask, next_seq_tokens


def build_pyramid_schedule(
    initial_seq_tokens: int,
    num_ns_tokens: int,
    num_layers: int,
    final_seq_query_tokens: int,
    round_to: int,
    use_pyramid: bool,
) -> list[int]:
    if not use_pyramid or initial_seq_tokens <= final_seq_query_tokens:
        return [initial_seq_tokens + num_ns_tokens] * num_layers

    final_seq_query_tokens = max(0, min(initial_seq_tokens, final_seq_query_tokens))
    if num_layers == 1:
        return [final_seq_query_tokens + num_ns_tokens]

    schedule = []
    previous = initial_seq_tokens
    for layer_index in range(num_layers):
        if layer_index == num_layers - 1:
            current = final_seq_query_tokens
        else:
            ratio = layer_index / (num_layers - 1)
            target = initial_seq_tokens + (final_seq_query_tokens - initial_seq_tokens) * ratio
            current = int(round(target))
            current = max(final_seq_query_tokens, current)
            if round_to > 1:
                current = max(final_seq_query_tokens, int(round(current / round_to) * round_to))
        current = min(previous, current)
        schedule.append(current + num_ns_tokens)
        previous = current
    return schedule
