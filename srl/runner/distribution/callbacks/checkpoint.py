import datetime
import logging
import os
import time
from dataclasses import dataclass

from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.task_manager import TaskManager

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(DistributionCallback, Evaluate):
    save_dir: str = "checkpoints"
    interval: int = 60 * 20  # s

    def on_start(self, task_manager: TaskManager):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")
        self._eval_runner = None
        self.interval_t0 = time.time()
        self._save_parameter(task_manager, is_last=False)

    def on_polling(self, task_manager: TaskManager):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(task_manager, is_last=False)
            self.interval_t0 = time.time()

    def on_end(self, task_manager: TaskManager):
        self._save_parameter(task_manager, is_last=True)

    def _save_parameter(self, task_manager: TaskManager, is_last: bool):
        task_config = task_manager.get_config()
        if task_config is None:
            return
        parameter = task_manager.create_parameter()
        if parameter is None:
            return

        try:
            if self._eval_runner is None:
                self._eval_runner = self.create_eval_runner(task_config.context)
            eval_rewards = self.run_eval(self._eval_runner, parameter)
            if eval_rewards is None:
                eval_rewards = "None"
        except Exception:
            import traceback

            logger.error(traceback.format_exc())
            eval_rewards = "None"

        fn = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn += f"_{eval_rewards}"
        if is_last:
            fn += "_last"
        fn += ".pickle"

        parameter.save(os.path.join(self.save_dir, fn))
