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

        self.interval_t0 = time.time()
        self._save_parameter(task_manager, is_last=False)

    def on_polling(self, task_manager: TaskManager):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(task_manager, is_last=False)
            self.interval_t0 = time.time()

    def on_end(self, task_manager: TaskManager):
        self._save_parameter(task_manager, is_last=True)

    def _save_parameter(self, task_manager: TaskManager, is_last: bool):
        if self.runner is None:
            self.runner = task_manager.create_runner()
        if self.runner is None:
            return

        parameter = self.runner.make_parameter(is_load=False)
        task_manager.read_parameter(parameter)
        
        if self.setup_eval_runner(self.runner):
            eval_rewards = self.run_eval(parameter)
        else:
            eval_rewards = "None"

        fn = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn += f"_{eval_rewards}"
        if is_last:
            fn += "_last"
        fn += ".pickle"

        parameter.save(os.path.join(self.save_dir, fn))
