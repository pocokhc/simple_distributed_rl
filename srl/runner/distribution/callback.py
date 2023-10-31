from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from srl.runner.distribution.manager import DistributedManager
    from srl.runner.runner import Runner


class DistributionCallback(ABC):
    def on_start(self, runner: "Runner", manager: "DistributedManager", task_id: str) -> None:
        pass  # do nothing

    def on_polling(self, runner: "Runner", manager: "DistributedManager", task_id: str) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False

    def on_end(self, runner: "Runner", manager: "DistributedManager", task_id: str) -> None:
        pass  # do nothing


class ActorServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False


class TrainerServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False
