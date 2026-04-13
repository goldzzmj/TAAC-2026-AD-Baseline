from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.metrics import classification_metrics


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    output_dir: str = "outputs/onetrans_small_demo"


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_examples = 0
    logits_buffer: list[np.ndarray] = []
    labels_buffer: list[np.ndarray] = []
    binary_labels_buffer: list[np.ndarray] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["labels"]
        binary_labels = batch["binary_labels"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        logits_buffer.append(logits.detach().cpu().numpy())
        labels_buffer.append(labels.detach().cpu().numpy())
        binary_labels_buffer.append(binary_labels.detach().cpu().numpy())

    merged_logits = np.concatenate(logits_buffer, axis=0)
    merged_labels = np.concatenate(labels_buffer, axis=0)
    merged_binary = np.concatenate(binary_labels_buffer, axis=0)
    metrics = classification_metrics(merged_logits, merged_labels, merged_binary)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = _move_batch_to_device(value, device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cfg: TrainerConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_metrics: dict[str, Any] = {}
    best_checkpoint = checkpoint_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics = _run_epoch(model, valid_loader, criterion, device, optimizer=None)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(record)

        valid_auc = valid_metrics["auc"]
        if np.isnan(valid_auc):
            valid_auc = float("-inf")
        if valid_auc >= best_auc:
            best_auc = valid_auc
            best_metrics = record
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "trainer_config": asdict(cfg),
                },
                best_checkpoint,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_metrics["loss"], 6),
                    "train_auc": train_metrics["auc"],
                    "valid_loss": round(valid_metrics["loss"], 6),
                    "valid_auc": valid_metrics["auc"],
                },
                ensure_ascii=True,
            )
        )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_checkpoint),
        "best_metrics": best_metrics,
        "history_path": str(history_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
