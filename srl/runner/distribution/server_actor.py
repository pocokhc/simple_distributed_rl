import datetime
import logging
import os
import pickle
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connector_configs import IMemoryServerParameters, RedisParameters
from srl.runner.distribution.server_manager import ServerManager, TaskManager

logger = logging.getLogger(__name__)


def _task_assign(task: TaskManager, keepalive_threshold: int) -> Tuple[bool, int]:
    if task.get_status() != "ACTIVE":
        return False, 0

    # --- queue が setup されてから実行する
    if not task.is_setup_memory():
        return False, 0

    # --- task_cfgが作れるかcheck
    task_cfg = task.get_config()
    if task_cfg is None:
        return False, 0

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    for i in range(task.get_actor_num()):
        _aid = task.get_actor(i, "id")

        # --- healthチェック ---
        # 前回のactorが残っていたら削除
        _a_elapsed_time = (now_utc - task.get_actor_update_time(i)).total_seconds()
        if _a_elapsed_time > keepalive_threshold:
            task.unassign("actor", i, reason=f"(health time over) {_a_elapsed_time:.1f}s {_aid}")
        # ---------------------

        _aid = task.get_actor(i, "id")
        if _aid != "":
            continue

        # assign
        if task.assign(i):
            return True, i
        else:
            return False, 0

    return False, 0


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    callbacks: List[ActorServerCallback] = [],
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
):
    server = ServerManager(redis_params, memory_params)
    task_manager = server.get_task_manager("actor")
    redis_connector = server.get_redis_connector()
    memory_sender = server.get_memory_sender()

    procs = []
    try:
        print(f"wait actor: {task_manager.uid}")
        logger.info(f"wait actor: {task_manager.uid}")
        while True:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- server check
            if not redis_connector.ping():
                logger.info("Redis server connect fail.")
                time.sleep(10)
                continue
            if not memory_sender.ping():
                logger.info("MemorySender server connect fail.")
                time.sleep(10)
                continue

            # --- task check
            flag, actor_id = _task_assign(task_manager, keepalive_threshold)
            if flag:
                print(f"actor{task_manager.uid} start, actor_id={actor_id}")
                logger.info(f"actor{task_manager.uid} start, actor_id={actor_id}")

                # --- proc
                proc = subprocess.Popen(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "server_actor_run.py")],
                    stdin=subprocess.PIPE,
                    stdout=None,  # 出力は親のターミナルに流す
                )
                procs.append(proc)
                assert proc.stdin is not None
                proc.stdin.write(
                    pickle.dumps(
                        [
                            redis_params,
                            memory_params,
                            actor_id,
                            task_manager.uid,
                            keepalive_interval,
                        ]
                    )
                )
                proc.stdin.close()

    finally:
        logger.info("subprocess closing.")
        for proc in procs:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            finally:
                proc.kill()
