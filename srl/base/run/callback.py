from abc import ABC
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from srl.base.context import RunContext

    from .core_play import RunStateActor
    from .core_train_only import RunStateTrainer


class PlayCallback(ABC):
    def on_start(self, context: "RunContext", **kwargs) -> None:
        pass  # do nothing

    def on_end(self, context: "RunContext", **kwargs) -> None:
        pass  # do nothing


class RunCallback(PlayCallback, ABC):
    def on_episodes_begin(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
        pass  # do nothing

    def on_episodes_end(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
        pass  # do nothing

    # --- 実装されている場合に実行
    # def on_episode_begin(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing

    # def on_episode_end(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing

    # def on_step_action_before(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing

    # def on_step_action_after(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing

    # def on_step_begin(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing

    # def on_step_end(self, context: "RunContext", state: "RunStateActor", **kwargs) -> Optional[bool]:
    #    """If return is True, it will end intermediate stop."""
    #    return False

    # def on_skip_step(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
    #    pass  # do nothing


class TrainCallback(PlayCallback, ABC):
    def on_trainer_start(self, context: "RunContext", state: "RunStateTrainer", **kwargs) -> None:
        pass  # do nothing

    def on_trainer_end(self, context: "RunContext", state: "RunStateTrainer", **kwargs) -> None:
        pass  # do nothing

    # --- 実装されている場合に実行
    # def on_train_before(self, context: "RunContext", state: "RunStateTrainer", **kwargs) -> None:
    #    pass  # do nothing

    # def on_train_after(self, context: "RunContext", state: "RunStateTrainer", **kwargs) -> Optional[bool]:
    #    """If return is True, it will end intermediate stop."""
    #    return False


CallbackType = Union[RunCallback, TrainCallback]
