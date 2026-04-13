from __future__ import annotations

from typing import Callable, Dict, Type

from torch.utils.data import Dataset


DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str) -> Callable[[Type[Dataset]], Type[Dataset]]:
    def decorator(dataset_cls: Type[Dataset]) -> Type[Dataset]:
        DATASET_REGISTRY[name] = dataset_cls
        return dataset_cls

    return decorator


def dataset_factory(name: str) -> Type[Dataset]:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


from src.data_module.dataset.taac2026_demo_dataset import (  # noqa: E402
    FeatureSchema,
    TAAC2026DemoDataset,
    build_feature_schema,
    create_taac2026_collate_fn,
)


__all__ = [
    "DATASET_REGISTRY",
    "FeatureSchema",
    "TAAC2026DemoDataset",
    "build_feature_schema",
    "create_taac2026_collate_fn",
    "dataset_factory",
    "register_dataset",
]
