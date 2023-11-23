import logging
import time
import traceback
from typing import List

import srl
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskManagerParams

logger = logging.getLogger(__name__)


def run(
    runner: srl.Runner,
    redis_params: RedisParameters,
    wait: bool,
    parameter_sync_interval: int = 60,
    callbacks: List[DistributionCallback] = [],
):
    task_manager_params = TaskManagerParams("client")
    manager = ServerManager(redis_params, None, task_manager_params)

    task_manager = manager.get_task_manager()
    task_manager.create(runner.create_task_config(), runner.make_parameter().backup(to_cpu=True))

    if not wait:
        return

    try:
        # callbacks
        [c.on_start(runner, manager) for c in callbacks]

        parameter_reader = manager.get_parameter_reader()

        # --- run
        print("wait task")
        t0 = time.time()
        while True:
            try:
                time.sleep(1)

                # --- param sync
                if time.time() - t0 > parameter_sync_interval:
                    t0 = time.time()
                    try:
                        params = parameter_reader.parameter_read()
                        if params is not None:
                            runner.make_parameter().restore(params, from_cpu=True)
                    except Exception:
                        logger.warning(traceback.format_exc())

                # --- end check ---
                if task_manager.is_finished():
                    try:
                        params = parameter_reader.parameter_read()
                        if params is not None:
                            runner.make_parameter().restore(params, from_cpu=True)
                    except Exception:
                        logger.warning(traceback.format_exc())
                    finally:
                        break
                # -----------------

                _stop_flags = [c.on_polling(runner, manager) for c in callbacks]
                if True in _stop_flags:
                    break

            except Exception:
                logger.error(traceback.format_exc())

        [c.on_end(runner, manager) for c in callbacks]
    finally:
        task_manager.finished("client end")
        print("task end")
        logger.info("task end")
