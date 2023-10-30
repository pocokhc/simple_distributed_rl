from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .context import RunContext
    from .core import RunState


class CallbackData:
    def set_data(self, context: "RunContext", state: "RunState"):
        self.context = context
        self.state = state


class Callback(ABC):
    def on_episodes_begin(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_episodes_end(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_episode_begin(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_episode_end(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_step_action_before(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_step_begin(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_step_end(self, dat: CallbackData) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False  # do nothing

    def on_skip_step(self, dat: CallbackData) -> None:
        pass  # do nothing


class TrainerCallback(ABC):
    def on_trainer_start(self, dat: CallbackData) -> None:
        pass  # do nothing

    def on_trainer_loop(self, dat: CallbackData) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False  # do nothing

    def on_trainer_end(self, dat: CallbackData) -> None:
        pass  # do nothing
