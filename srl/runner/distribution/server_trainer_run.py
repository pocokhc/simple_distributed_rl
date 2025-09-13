import logging
import pickle
import sys
import threading
import time
import traceback

from srl.base.context import RunContext
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_train_only
from srl.base.run.callback import RunCallback
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.distribution.server_manager import ServerManager

logger = logging.getLogger(__name__)


class _ShareData:
    def __init__(self):
        self.sync_count = 0
        self.train_count = 0
        self.q_recv_count = 0
        self.end_signal = False
        self.th_error = ""


def _memory_communicate(
    server: ServerManager,
    memory: RLMemory,
    share_data: _ShareData,
):
    try:
        memory_receiver = server.get_memory_receiver()
        memory_receiver.ping()

        worker_funcs = {k: (v[0], v[1] is None) for k, v in memory.get_worker_funcs().items()}
        while not share_data.end_signal:
            if not memory_receiver.is_connected:
                time.sleep(1)
                if not memory_receiver.ping():
                    continue

            try:
                dat = memory_receiver.memory_recv()
            except Exception as e:
                logger.error(f"Memory recv error: {e}")
                continue

            if dat is None:
                time.sleep(0.1)
            else:
                name, raw = dat
                if worker_funcs[name][1]:
                    args, kwargs = pickle.loads(raw)
                    worker_funcs[name][0](*args, **kwargs)
                else:
                    worker_funcs[name][0](*raw, serialized=True)
                share_data.q_recv_count += 1

    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
        logger.info("trainer memory thread end.")


def _parameter_communicate(
    server: ServerManager,
    parameter: RLParameter,
    share_data: _ShareData,
    trainer_parameter_send_interval: int,
    uid: str,
    keepalive_interval: int,
):
    try:
        # --- parameter
        parameter_writer = server.get_parameter_server()
        parameter_writer.ping()
        keepalive_t0 = 0

        # --- task keepalive
        task = server.get_task_manager("trainer", uid=uid)
        start_train_count = task.get_train_count()

        while not share_data.end_signal:
            time.sleep(trainer_parameter_send_interval)
            params = parameter.backup(serialized=True)
            if params is not None:
                parameter_writer.parameter_write(params)
                share_data.sync_count += 1

            # --- keepalive
            if time.time() - keepalive_t0 > keepalive_interval:
                if task.is_finished():
                    break
                task.keepalive_trainer(start_train_count, share_data)
                keepalive_t0 = time.time()

        task.keepalive_trainer(start_train_count, share_data)
    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
        logger.info("trainer parameter thread end.")


class _TrainerInterruptThread(RunCallback):
    def __init__(
        self,
        memory_th: threading.Thread,
        parameter_th: threading.Thread,
        share_data: _ShareData,
    ) -> None:
        self.memory_th = memory_th
        self.parameter_th = parameter_th
        self.share_data = share_data
        self.t0_health = time.time()

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        if not state.is_step_trained:
            time.sleep(1)  # warmupなら待機
        self.share_data.train_count = state.trainer.get_train_count()
        state.sync_trainer = self.share_data.sync_count
        state.trainer_recv_q = self.share_data.q_recv_count

        if time.time() - self.t0_health > 5:
            if not self.memory_th.is_alive():
                self.share_data.end_signal = True
            if not self.parameter_th.is_alive():
                self.share_data.end_signal = True
            self.t0_health = time.time()
        return self.share_data.end_signal


def run_trainer(server: ServerManager, uid: str, keepalive_interval: int):
    task = server.get_task_manager("trainer", uid=uid)
    memory_th = None
    parameter_th = None
    share_data = _ShareData()
    try:
        task_cfg = task.get_config()
        if task_cfg is None:
            return

        context = task_cfg.context
        context.run_name = "trainer"
        context.setup_device()
        context.training = True
        context.distributed = True
        context.train_only = False

        # --- parameter
        parameter = context.rl_config.make_parameter()
        task.read_parameter(parameter)
        task.write_parameter(parameter)

        # --- memory
        memory = context.rl_config.make_memory()

        memory_th = threading.Thread(
            target=_memory_communicate,
            args=(
                server,
                memory,
                share_data,
            ),
        )
        parameter_th = threading.Thread(
            target=_parameter_communicate,
            args=(
                server,
                parameter,
                share_data,
                task_cfg.trainer_parameter_send_interval,
                uid,
                keepalive_interval,
            ),
        )
        memory_th.start()
        parameter_th.start()

        # --- play
        context.callbacks.append(
            _TrainerInterruptThread(
                memory_th,
                parameter_th,
                share_data,
            )
        )
        trainer = context.rl_config.make_trainer(parameter, memory)
        core_train_only.play_trainer_only(context, trainer)

        task.finished("trainer end")
        task.unassign("trainer", reason="completed")
        print("train end")
    finally:
        share_data.end_signal = True

        # --- last params
        task.write_parameter(parameter)
        print("send last parameter")

        # --- thread
        if memory_th is not None:
            memory_th.join(timeout=10)
        if parameter_th is not None:
            parameter_th.join(timeout=10)

        if share_data.th_error != "":
            raise ValueError(share_data.th_error)


if __name__ == "__main__":
    redis_params, memory_params, uid, keepalive_interval = pickle.loads(sys.stdin.buffer.read())
    print("--- start train process")
    server = ServerManager(redis_params, memory_params)
    try:
        run_trainer(server, uid, keepalive_interval)
    except Exception:
        print(traceback.format_exc())
    finally:
        print("--- end train process")
