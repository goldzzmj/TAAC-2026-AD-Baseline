from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_module.dataset import (  # noqa: E402
    TAAC2026DemoDataset,
    create_taac2026_collate_fn,
)
from src.model_module.model import OneTransSmall, OneTransSmallConfig  # noqa: E402
from src.trainer_module.onetrans_trainer import TrainerConfig, fit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a OneTrans-small style baseline.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo_1000_train.parquet",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo_1000_test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "onetrans_small_demo",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hash-bucket-size", type=int, default=50000)
    parser.add_argument("--max-list-length", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--positive-label-value", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataset = TAAC2026DemoDataset(
        parquet_path=args.train_path,
        positive_label_value=args.positive_label_value,
    )
    valid_dataset = TAAC2026DemoDataset(
        parquet_path=args.valid_path,
        schema=train_dataset.schema,
    )

    collate_fn = create_taac2026_collate_fn(
        schema=train_dataset.schema,
        max_list_length=args.max_list_length,
        max_seq_length=args.max_seq_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model_cfg = OneTransSmallConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        hash_bucket_size=args.hash_bucket_size,
        max_seq_length=args.max_seq_length,
    )
    trainer_cfg = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=str(args.output_dir),
    )
    model = OneTransSmall(schema=train_dataset.schema, cfg=model_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": train_dataset.schema.to_dict(),
        "model_config": asdict(model_cfg),
        "trainer_config": asdict(trainer_cfg),
        "label_to_index": train_dataset.label_to_index,
        "index_to_label": train_dataset.index_to_label,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        cfg=trainer_cfg,
        metadata=metadata,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
