from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srl.base.context import RunContext

    from .core_play import RunStateActor
    from .core_train_only import RunStateTrainer


class RunCallback(ABC):
    def on_start(self, context: "RunContext", **kwargs) -> None:
        pass  # do nothing

    def on_end(self, context: "RunContext", **kwargs) -> None:
        pass  # do nothing

    # ----------------------------------
    # worker
    # ----------------------------------
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

    # ----------------------------------
    # trainer
    # ----------------------------------
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

    # ----------------------------------
    # memory
    # ----------------------------------
    def on_memory_start(self, context: "RunContext", info: dict, **kwargs) -> None:
        pass  # do nothing

    def on_memory_end(self, context: "RunContext", info: dict, **kwargs) -> None:
        pass  # do nothing

    # --- 実装されている場合に実行
    # def on_memory(self, context: "RunContext", info: dict, **kwargs) -> None:
    #    pass  # do nothing
