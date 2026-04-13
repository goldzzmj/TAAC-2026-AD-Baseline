from __future__ import annotations

from typing import Callable, Dict, Type

from torch import nn


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = model_cls
        return model_cls

    return decorator


def model_factory(name: str) -> Type[nn.Module]:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


from src.model_module.model.onetrans import OneTrans, OneTransBackbone  # noqa: E402
from src.model_module.model.schema import (  # noqa: E402
    ModelConfig,
    NonSequentialFeatureSpec,
    SequentialFeatureSpec,
)


__all__ = [
    "MODEL_REGISTRY",
    "ModelConfig",
    "NonSequentialFeatureSpec",
    "OneTrans",
    "OneTransBackbone",
    "SequentialFeatureSpec",
    "model_factory",
    "register_model",
]
