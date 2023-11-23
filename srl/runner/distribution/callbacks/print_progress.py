import datetime
import logging
import time
from dataclasses import dataclass

import numpy as np

import srl
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.task_manager import TaskManager
from srl.runner.runner import TaskConfig
from srl.utils.util_str import to_str_reward, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class PrintProgress(DistributionCallback, Evaluate):
    interval: int = 60
    eval_worker: int = 0

    def __post_init__(self):
        self._eval_runner = None
        self._task_config = None

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.interval:
            return False
        self.progress_t0 = _time

        return True

    def _eval_str(self, task_manager: TaskManager, task_config: TaskConfig) -> str:
        if self._eval_runner is None:
            runner = srl.Runner(
                task_config.context.env_config,
                task_config.context.rl_config,
                task_config.config,
                task_config.context,
            )
            self.setup_eval_runner(runner)
        if self._eval_runner is None:
            return ""

        parameter = self._eval_runner.make_parameter(is_load=False)
        if not task_manager.read_parameter(parameter):
            return ""

        eval_rewards = self._eval_runner.callback_play_eval(parameter)
        eval_rewards = np.mean(eval_rewards, axis=0)
        return f"({to_str_reward(eval_rewards[self.eval_worker])}eval)"

    # -----------------------------------------------------

    def on_start(self, task_manager: TaskManager):
        _time = time.time()
        self.progress_t0 = 0
        self.elapsed_t0 = _time

        self.t0_train_time = _time
        self.t0_train_count = 0

    def on_end(self, task_manager: TaskManager):
        self._print(task_manager)

    def on_polling(self, task_manager: TaskManager):
        if self._check_print_progress():
            self._print(task_manager)

    # -----------------------------------------

    def _print(self, task_manager: TaskManager):
        if self._task_config is None:
            self._task_config = task_manager.get_config()
        if self._task_config is None:
            print("Failed to get Task config.")
            return

        context = self._task_config.context
        _time = time.time()
        elapsed_time = _time - self.elapsed_t0
        actor_num = self._task_config.context.actor_num
        status = task_manager.get_status()

        # [TIME] [status] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        s += f" {status} {to_str_time(elapsed_time)}"

        # calc time
        train_count = task_manager.get_train_count()
        if train_count == "":
            train_count = 0
        else:
            train_count = int(train_count)
        _d = train_count - self.t0_train_count
        if _d > 0:
            train_time = (_time - self.t0_train_time) / _d
        else:
            train_time = np.inf
        self.t0_train_time = _time
        self.t0_train_count = train_count

        # [remain]
        if (context.max_train_count > 0) and (train_count > 0):
            remain_train = (context.max_train_count - train_count) * train_time
        else:
            remain_train = np.inf
        if context.timeout > 0:
            remain_time = context.timeout - elapsed_time
        else:
            remain_time = np.inf
        remain = min(remain_train, remain_time)
        if remain == np.inf:
            s += "(     - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [eval]
        if self.enable_eval:
            s += self._eval_str(task_manager, self._task_config)

        print(s)

        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # --- trainer
        trainer_id = task_manager.get_trainer("id")
        if trainer_id == "":
            print(" trainer  not assigned")
        else:
            _elapsed_time = (now_utc - task_manager.get_trainer_update_time()).total_seconds()
            s = f" trainer  {trainer_id} {_elapsed_time:.1f}s: "
            s += f" {train_count:5d}tr"
            s += f", {train_time:.3f}s/tr"

            memory_size = task_manager.get_trainer("memory")
            s += f", {memory_size}mem"

            print(s)

        # --- actor
        for idx in range(actor_num):
            aid = task_manager.get_actor(idx, "id")
            if aid == "":
                print(f" actor{idx:<3d} not assigned")
            else:
                _elapsed_time = (now_utc - task_manager.get_actor_update_time(idx)).total_seconds()
                s = task_manager.get_actor(idx, "episode")
                print(f" actor{idx:<3d} {aid} {_elapsed_time:.1f}s: {s:>6s}ep")
