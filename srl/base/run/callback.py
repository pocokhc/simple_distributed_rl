from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srl.base.context import RunContext

    from .core_play import RunStateActor
    from .core_train_only import RunStateTrainer


class RunCallback(ABC):
    def on_episodes_begin(self, context: "RunContext", state: "RunStateActor") -> None:
        pass  # do nothing

    def on_episodes_end(self, context: "RunContext", state: "RunStateActor") -> None:
        pass  # do nothing

    # --- 実装されている場合に実行
    # def on_episode_begin(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing

    # def on_episode_end(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing

    # def on_step_action_before(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing

    # def on_step_action_after(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing

    # def on_step_begin(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing

    # def on_step_end(self, context: "RunContext", state: "RunStateActor") -> Optional[bool]:
    #    """If return is True, it will end intermediate stop."""
    #    return False

    # def on_skip_step(self, context: "RunContext", state: "RunStateActor") -> None:
    #    pass  # do nothing


class TrainerCallback(ABC):
    def on_trainer_start(self, context: "RunContext", state: "RunStateTrainer") -> None:
        pass  # do nothing

    # --- 実装されている場合に実行
    # def on_trainer_loop(self, context: "RunContext", state: "RunStateTrainer") -> Optional[bool]:
    #    """If return is True, it will end intermediate stop."""
    #    return False

    def on_trainer_end(self, context: "RunContext", state: "RunStateTrainer") -> None:
        pass  # do nothing
