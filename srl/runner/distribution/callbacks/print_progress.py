import datetime
import logging
import time
from dataclasses import dataclass

import numpy as np

from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.runner import Runner
from srl.utils.util_str import to_str_reward, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class PrintProgress(DistributionCallback, Evaluate):
    interval: int = 60
    eval_worker: int = 0

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.interval:
            return False
        self.progress_t0 = _time

        return True

    def _eval_str(self, runner: Runner) -> str:
        if self.eval_runner is None:
            return ""
        parameter = runner.make_parameter(is_load=False)
        if runner.context.distributed:
            if runner.context.actor_id == 0:
                eval_rewards = self.run_eval(parameter)
                return f"({to_str_reward(eval_rewards[self.eval_worker])}eval)"
            else:
                return " " * 12
        else:
            eval_rewards = self.run_eval(parameter)
            return f"({to_str_reward(eval_rewards[self.eval_worker])}eval)"

    # -----------------------------------------------------

    def on_start(self, runner: Runner, manager: ServerManager):
        context = runner.context
        self.setup_eval_runner(runner)

        s = f"### env: {runner.env_config.name}, rl: {runner.rl_config.getName()}"
        if context.max_episodes > 0:
            s += f", max episodes: {context.max_episodes}"
        if context.timeout > 0:
            s += f", timeout: {to_str_time(context.timeout)}"
        if context.max_steps > 0:
            s += f", max steps: {context.max_steps}"
        if context.max_train_count > 0:
            s += f", max train: {context.max_train_count}"
        if context.max_memory > 0:
            s += f", max memory: {context.max_memory}"
        print(s)

        _time = time.time()
        self.progress_t0 = 0
        self.elapsed_t0 = _time

        self.t0_train_time = _time
        self.t0_train_count = 0

    def on_end(self, runner: Runner, manager: ServerManager):
        self._print(runner, manager)

    def on_polling(self, runner: Runner, manager: ServerManager):
        if self._check_print_progress():
            self._print(runner, manager)

    # -----------------------------------------

    def _print(self, runner: Runner, manager: ServerManager):
        context = runner.context
        _time = time.time()
        elapsed_time = _time - self.elapsed_t0
        task_manager = manager.get_task_manager()

        actor_num = runner.context.actor_num
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
        s += self._eval_str(runner)

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
