from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .runner import Runner


class RunnerCallback(ABC):
    def __init__(self) -> None:
        self.runner: Optional["Runner"] = None

    def on_runner_start(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_runner_end(self, runner: "Runner") -> None:
        pass  # do nothing
