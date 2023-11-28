import logging
from typing import TYPE_CHECKING, Optional

from srl.runner.distribution.interface import (
    IMemoryReceiver,
    IMemorySender,
    IMemoryServerParameters,
    IParameterReader,
    IParameterWriter,
)
from srl.runner.distribution.task_manager import TaskManager, TaskManagerParams

if TYPE_CHECKING:
    from srl.runner.distribution.connectors.parameters import RedisParameters
    from srl.runner.distribution.connectors.redis_ import RedisConnector

logger = logging.getLogger(__name__)


class ServerManager:
    def __init__(
        self,
        redis_params: "RedisParameters",
        memory_params: Optional[IMemoryServerParameters],
        task_manager_params: TaskManagerParams,
    ):
        self.redis_params = redis_params
        self.memory_params = memory_params

        self._task_manager_params = task_manager_params
        self._task_manager_params.task_name = redis_params.task_name
        self._task_manager = None

        self._redis_connector = None
        self._memory_receiver = None
        self._memory_sender = None

    def _copy_args(self):
        return (
            self.redis_params,
            self.memory_params,
            self.get_task_manager().params,
        )

    @staticmethod
    def _copy(
        redis_params,
        memory_params,
        task_manager_params,
    ):
        return ServerManager(redis_params, memory_params, task_manager_params)

    def get_redis_connector(self) -> "RedisConnector":
        if self._redis_connector is None:
            self._redis_connector = self.redis_params.create_connector()
        return self._redis_connector

    def get_task_manager(self) -> TaskManager:
        if self._task_manager is None:
            self._task_manager = TaskManager.new_connector(self.get_redis_connector(), self._task_manager_params)
        return self._task_manager

    def get_parameter_writer(self) -> IParameterWriter:
        return self.get_redis_connector()

    def get_parameter_reader(self) -> IParameterReader:
        return self.get_redis_connector()

    def get_memory_receiver(self) -> IMemoryReceiver:
        if self.memory_params is None:
            return self.get_redis_connector()
        if self._memory_receiver is None:
            self._memory_receiver = self.memory_params.create_memory_receiver()
        return self._memory_receiver

    def get_memory_sender(self) -> IMemorySender:
        if self.memory_params is None:
            return self.get_redis_connector()
        if self._memory_sender is None:
            self._memory_sender = self.memory_params.create_memory_sender()
        return self._memory_sender
