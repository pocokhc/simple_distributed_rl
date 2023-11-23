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

    def on_base_run_start(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_base_run_end(self, runner: "Runner") -> None:
        pass  # do nothing


class GameCallback(ABC):
    def on_game_init(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_game_begin(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_game_end(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_game_step_end(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_skip_step(self, runner: "Runner") -> None:
        pass  # do nothing
