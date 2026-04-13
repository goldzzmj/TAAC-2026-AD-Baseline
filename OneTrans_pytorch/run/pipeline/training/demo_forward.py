from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.model_module.model import ModelConfig, NonSequentialFeatureSpec, OneTrans, SequentialFeatureSpec  # noqa: E402


def build_dummy_batch(batch_size: int = 2) -> dict[str, dict[str, torch.Tensor]]:
    seq_length = 6
    return {
        "non_sequential": {
            "user_profile": torch.randn(batch_size, 32),
            "item_context": torch.randn(batch_size, 48),
            "request_context": torch.randn(batch_size, 16),
        },
        "sequential": {
            "click_seq": {
                "values": torch.randn(batch_size, seq_length, 40),
                "mask": torch.tensor(
                    [[True, True, True, True, True, False], [True, True, True, False, False, False]]
                ),
                "timestamps": torch.tensor(
                    [[1, 3, 5, 7, 9, 0], [2, 4, 6, 0, 0, 0]],
                    dtype=torch.long,
                ),
            },
            "cart_seq": {
                "values": torch.randn(batch_size, seq_length, 24),
                "mask": torch.tensor(
                    [[True, True, False, False, False, False], [True, True, True, True, False, False]]
                ),
                "timestamps": torch.tensor(
                    [[2, 8, 0, 0, 0, 0], [1, 5, 7, 11, 0, 0]],
                    dtype=torch.long,
                ),
            },
            "order_seq": {
                "values": torch.randn(batch_size, seq_length, 20),
                "mask": torch.tensor(
                    [[False, False, False, False, False, False], [True, False, False, False, False, False]]
                ),
                "timestamps": torch.tensor(
                    [[0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0]],
                    dtype=torch.long,
                ),
            },
        },
    }


def main() -> None:
    cfg = ModelConfig(
        non_sequential_features=(
            NonSequentialFeatureSpec(name="user_profile", input_dim=32),
            NonSequentialFeatureSpec(name="item_context", input_dim=48),
            NonSequentialFeatureSpec(name="request_context", input_dim=16),
        ),
        sequential_features=(
            SequentialFeatureSpec(name="click_seq", input_dim=40, impact_order=1),
            SequentialFeatureSpec(name="cart_seq", input_dim=24, impact_order=2),
            SequentialFeatureSpec(name="order_seq", input_dim=20, impact_order=3),
        ),
        hidden_dim=256,
        num_heads=4,
        num_layers=6,
        ffn_hidden_dim=1024,
        ns_tokenizer_type="auto_split",
        ns_token_count=12,
        sequence_merge_type="timestamp_aware",
        output_dim=1,
    )
    model = OneTrans(cfg)
    batch = build_dummy_batch()
    outputs = model(batch, labels=torch.tensor([0.0, 1.0]))

    print("logits:", outputs["logits"].shape)
    print("all tokens:", outputs["tokens"].shape)
    print("sequence tokens:", outputs["sequence_tokens"].shape)
    print("non-sequential tokens:", outputs["non_sequential_tokens"].shape)
    print("loss:", float(outputs["loss"]))


if __name__ == "__main__":
    main()
