from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from srl.runner.distribution.task_manager import TaskManager


class DistributionCallback(ABC):
    def on_start(self, task_manager: "TaskManager") -> None:
        pass  # do nothing

    def on_polling(self, task_manager: "TaskManager") -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False

    def on_end(self, task_manager: "TaskManager") -> None:
        pass  # do nothing


class ActorServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False


class TrainerServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False
