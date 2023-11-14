import logging
import time
import traceback
from typing import List, Optional

import srl
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.manager import DistributedManager

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, redis_parameter: RedisParameters) -> None:
        self.manager = DistributedManager(redis_parameter, None)

    def read_status(self) -> str:
        assert self.manager.ping()
        return self.manager.task_get_status()

    def read_parameter(self):
        assert self.manager.ping()
        return self.manager.parameter_read()

    def create_task_runner(self) -> Optional[srl.Runner]:
        assert self.manager.ping()
        task_config = self.manager.task_get_config()
        if task_config is None:
            return None
        runner = srl.Runner(
            task_config.context.env_config,
            task_config.context.rl_config,
            task_config.config,
            task_config.context,
        )
        params = self.manager.parameter_read()
        if params is not None:
            runner.make_parameter().restore(params)
        return runner


def run(
    runner: srl.Runner,
    redis_parameter: RedisParameters,
    wait: bool,
    callbacks: List[DistributionCallback] = [],
):
    parameter = runner.make_parameter()
    manager = DistributedManager(redis_parameter, None)
    assert manager.ping()
    manager.set_user("client")

    manager.task_create(
        runner.context.actor_num,
        runner.create_task_config(["Checkpoint", "HistoryOnFile", "HistoryOnMemory"]),
        parameter.backup(to_cpu=True),
    )

    if not wait:
        return

    try:
        # callbacks
        [c.on_start(runner, manager) for c in callbacks]

        # --- run
        print("wait task")
        while True:
            try:
                time.sleep(1)
                if manager.keepalive_client():
                    # 定期的に同期する
                    try:
                        params = manager.parameter_read()
                        if params is not None:
                            runner.make_parameter().restore(params, from_cpu=True)
                    except Exception:
                        logger.warning(traceback.format_exc())

                # --- end check ---
                if manager.task_get_status() == "END":
                    try:
                        params = manager.parameter_read()
                        if params is not None:
                            runner.make_parameter().restore(params, from_cpu=True)
                    except Exception:
                        logger.warning(traceback.format_exc())
                    finally:
                        print("task end")
                        logger.info("task end")
                        break
                # -----------------

                _stop_flags = [c.on_polling(runner, manager) for c in callbacks]
                if True in _stop_flags:
                    break

            except Exception:
                logger.error(traceback.format_exc())

        [c.on_end(runner, manager) for c in callbacks]

    finally:
        manager.task_end()
