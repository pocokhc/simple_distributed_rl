import datetime
import logging
import os
import time
from dataclasses import dataclass

from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(DistributionCallback, Evaluate):
    save_dir: str = "checkpoints"
    interval: int = 60 * 20  # s

    def on_start(self, runner: Runner, manager: DistributedManager, task_id: str):
        self.setup_eval_runner(runner)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        self.interval_t0 = time.time()
        self._save_parameter(runner, manager, task_id, is_last=True)

    def on_polling(self, runner: Runner, manager: DistributedManager, task_id: str):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(runner, manager, task_id, is_last=False)
            self.interval_t0 = time.time()

    def on_end(self, runner: Runner, manager: DistributedManager, task_id: str):
        self._save_parameter(runner, manager, task_id, is_last=True)

    def _save_parameter(self, runner: Runner, manager: DistributedManager, task_id: str, is_last: bool):
        parameter = runner.make_parameter(is_load=False)

        params = manager.parameter_read(task_id)
        if params is not None:
            parameter.restore(params)

        if self.enable_eval:
            eval_rewards = self.run_eval(runner.state.parameter)
        else:
            eval_rewards = "None"

        fn = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn += f"_{eval_rewards}"
        if is_last:
            fn += "_last"
        fn += ".pickle"

        parameter.save(os.path.join(self.save_dir, fn))
