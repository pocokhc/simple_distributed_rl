from abc import ABC
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .runner import Runner


class Callback(ABC):
    def on_episodes_begin(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_episodes_end(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_episode_begin(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_episode_end(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_step_action_before(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_step_begin(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_step_end(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_skip_step(self, runner: "Runner") -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, runner: "Runner") -> bool:
        return False


class TrainerCallback(ABC):
    def on_trainer_start(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_trainer_train(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_trainer_end(self, runner: "Runner") -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, runner: "Runner") -> bool:
        return False


class MPCallback(ABC):
    def on_init(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_start(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_polling(self, runner: "Runner") -> None:
        pass  # do nothing

    def on_end(self, runner: "Runner") -> None:
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


CallbackType = Union[Callback, TrainerCallback, MPCallback, GameCallback]
