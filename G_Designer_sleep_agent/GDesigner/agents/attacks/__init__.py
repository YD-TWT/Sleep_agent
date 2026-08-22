from __future__ import annotations

from typing import Dict, Optional, Type

from GDesigner.agents.attacks.base import AttackStrategy
from GDesigner.agents.attacks.numeric_perturb import NumericPerturb
from GDesigner.agents.attacks.letter_perturb import (
    LetterPerturb,
    MMLUProLetterPerturb,
)

_STRATEGY_REGISTRY: Dict[str, Type[AttackStrategy]] = {
    NumericPerturb.name: NumericPerturb,
    LetterPerturb.name: LetterPerturb,
    MMLUProLetterPerturb.name: MMLUProLetterPerturb,
}

def list_strategies() -> list[str]:
    return sorted(_STRATEGY_REGISTRY.keys())

def resolve_strategy(
    name: Optional[str] = None,
    dataset: Optional[str] = None,
) -> AttackStrategy:
    if name:
        key = name.strip().lower()
        if key in _STRATEGY_REGISTRY:
            return _STRATEGY_REGISTRY[key]()
        raise ValueError(
            f"Unknown attack strategy '{name}'. "
            f"Available: {list_strategies()}"
        )

    if dataset:
        ds = dataset.strip().lower()
        for cls in _STRATEGY_REGISTRY.values():
            if ds in cls.applicable_datasets:
                return cls()

    return NumericPerturb()

__all__ = [
    "AttackStrategy",
    "NumericPerturb",
    "LetterPerturb",
    "MMLUProLetterPerturb",
    "resolve_strategy",
    "list_strategies",
]
