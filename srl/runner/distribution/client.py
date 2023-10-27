import logging
import time
import traceback

import srl
from srl.runner.distribution.manager import DistributedManager

logger = logging.getLogger(__name__)


def run(
    runner: srl.Runner,
    host: str,
    port: int = 6379,
    redis_kwargs: dict = {},
    keepalive_interval: int = 10,
):
    parameter = runner.make_parameter()

    manager = DistributedManager(host, port, redis_kwargs, keepalive_interval)
    manager.server_ping()
    manager.set_user("client")

    # --- task
    actor_num = runner.context.actor_num
    task_id = manager.task_add(
        actor_num,
        runner.create_task_config(),
        parameter.backup(),
    )

    # --- run
    _check_t0 = time.time()
    print(f"wait task: {task_id}")
    while True:
        try:
            time.sleep(1)
            manager.keepalive(task_id)

            # --- end check ---
            if manager.task_get_status(task_id) == "END":
                try:
                    params = manager.parameter_read()
                    if params is not None:
                        parameter.restore(params)
                except Exception:
                    logger.warning(traceback.format_exc())
                finally:
                    print(f"task end: {task_id}")
                    logger.info(f"task end: {task_id}")
                    break
            # -----------------

            # --- progress
            if time.time() - _check_t0 > keepalive_interval:
                _check_t0 = time.time()

                status = manager.task_get_status(task_id)
                qsize = manager.memory_size(task_id)
                print(f"--- {task_id} {status} {qsize}mem")

                # trainer
                tid = manager.task_get_trainer(task_id, "id")
                if tid == "":
                    print("trainer : not assigned")
                else:
                    s = manager.task_get_trainer(task_id, "train")
                    s2 = manager.task_get_trainer(task_id, "memory")
                    print(f"trainer : {tid} {s} train, {s2} memory")

                # actor
                for idx in range(actor_num):
                    aid = manager.task_get_actor(task_id, idx, "id")
                    if aid == "":
                        print(f"actor{idx:<3d}: not assigned")
                    else:
                        s = manager.task_get_actor(task_id, idx, "episode")
                        print(f"actor{idx:<3d}: {aid} {s} episode")

                # --- check param
                # 定期的にparameterを取得
                params = manager.parameter_read(task_id)
                if params is not None:
                    parameter.restore(params)
                    print(runner.evaluate())

        except Exception:
            logger.error(traceback.format_exc())
