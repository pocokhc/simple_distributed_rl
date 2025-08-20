from abc import ABC, abstractmethod
from dataclasses import dataclass, fields


@dataclass
class Scheduler(ABC):
    @classmethod
    def from_kwargs(cls, **kwargs):
        """不要な引数を無視してインスタンスを生成する"""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return cls(**filtered)

    @abstractmethod
    def update(self, step: int) -> "Scheduler":
        raise NotImplementedError()

    @abstractmethod
    def to_float(self) -> float:
        raise NotImplementedError()

    def __float__(self) -> float:
        return self.to_float()
