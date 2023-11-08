import logging
import time
import traceback
from typing import List

import srl
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.manager import DistributedManager

logger = logging.getLogger(__name__)


def run(
    runner: srl.Runner,
    redis_parameter: RedisParameters,
    callbacks: List[DistributionCallback] = [],
):
    parameter = runner.make_parameter()

    manager = DistributedManager(redis_parameter, None)
    assert manager.ping()
    manager.set_user("client")

    # --- task
    actor_num = runner.context.actor_num
    manager.task_create(
        actor_num,
        runner.create_task_config(["Checkpoint", "HistoryOnFile", "HistoryOnMemory"]),
        parameter.backup(to_cpu=True),
    )
    try:
        # callbacks
        [c.on_start(runner, manager) for c in callbacks]

        # --- run
        print("wait task")
        while True:
            try:
                time.sleep(1)
                if manager.keepalive():
                    # 定期的に同期する
                    try:
                        params = manager.parameter_read()
                        if params is not None:
                            parameter.restore(params, from_cpu=True)
                    except Exception:
                        logger.warning(traceback.format_exc())

                # --- end check ---
                if manager.task_get_status() == "END":
                    try:
                        params = manager.parameter_read()
                        if params is not None:
                            parameter.restore(params, from_cpu=True)
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
