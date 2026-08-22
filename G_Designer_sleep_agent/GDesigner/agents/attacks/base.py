from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

class AttackStrategy(ABC):

    name: ClassVar[str] = "base"
    applicable_datasets: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def apply(self, response: str, task: str) -> str:
        pass

    def __repr__(self) -> str:
        return f"<AttackStrategy {self.name}>"
