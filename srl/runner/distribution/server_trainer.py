import datetime
import logging
import os
import pickle
import subprocess
import sys
import time
from typing import List, Optional

from srl.runner.distribution.callback import TrainerServerCallback
from srl.runner.distribution.connector_configs import IMemoryServerParameters, RedisParameters
from srl.runner.distribution.server_manager import ServerManager, TaskManager

logger = logging.getLogger(__name__)


def _task_assign(task: TaskManager, keepalive_threshold: int) -> bool:
    if task.get_status() != "ACTIVE":
        return False

    # --- task_cfgが作れるかcheck
    task_cfg = task.get_config()
    if task_cfg is None:
        return False

    # --- アサイン時にhealthチェック ---
    # 前回のtrainerが残っていたら削除
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    _t_elapsed_time = (now_utc - task.get_trainer_update_time()).total_seconds()
    if _t_elapsed_time > keepalive_threshold:
        _tid = task.get_trainer("id")
        task.unassign("trainer", reason=f"(health check, time over) {_t_elapsed_time:.1f}s {_tid}")
    # ----------------------------------

    if task.get_trainer("id") != "":
        return False

    # assign
    if task.assign():
        return True
    else:
        return False


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    callbacks: List[TrainerServerCallback] = [],
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
):
    server = ServerManager(redis_params, memory_params)
    task_manager = server.get_task_manager("trainer")
    redis_connector = server.get_redis_connector()
    memory_receiver = server.get_memory_receiver()

    procs = []
    try:
        print(f"wait trainer: {task_manager.uid}")
        logger.info(f"wait trainer: {task_manager.uid}")
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
            if not memory_receiver.ping():
                logger.info("MemoryReceiver server connect fail.")
                time.sleep(10)
                continue

            # --- task assign
            if _task_assign(task_manager, keepalive_threshold):
                print(f"train run: {task_manager.uid}")
                logger.info(f"train run: {task_manager.uid}")

                # 最初にmemoryをリセット
                task_manager.setup_memory(memory_receiver, is_purge=True)

                # --- proc
                proc = subprocess.Popen(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "server_trainer_run.py")],
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
