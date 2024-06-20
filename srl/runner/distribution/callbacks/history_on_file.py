import datetime
import logging
import time
from dataclasses import dataclass

from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.callbacks.history_on_file import HistoryOnFileBase
from srl.runner.distribution.callback import DistributionCallback
from srl.runner.distribution.task_manager import TaskManager

logger = logging.getLogger(__name__)


@dataclass
class HistoryOnFile(DistributionCallback, Evaluate):
    save_dir: str = ""
    interval: int = 10  # s
    add_history: bool = False

    def __post_init__(self):
        self._base = HistoryOnFileBase(self.save_dir, self.add_history)

    def on_start(self, task_manager: TaskManager):
        self.task_config = task_manager.get_config()
        if self.task_config is not None:
            self._base.setup(self.task_config.context)

        self._base.open_fp("client", "client.txt")
        self.interval_t0 = time.time()

    def on_polling(self, task_manager: TaskManager):
        _time = time.time()
        if _time - self.interval_t0 > self.interval:
            self.interval_t0 = _time
            self._write_log(task_manager, is_last=False)

    def on_end(self, task_manager: TaskManager):
        self._write_log(task_manager, is_last=True)
        self._base.close()

    def _write_log(self, task_manager: TaskManager, is_last: bool):
        if not self._base.is_fp("client"):
            return
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        d = {
            "name": "client",
            "time": (now_utc - task_manager.get_create_time()).total_seconds(),
            "train": task_manager.get_train_count(),
        }

        if self.task_config is not None:
            parameter = task_manager.create_parameter()
            if parameter is not None:
                eval_rewards = self.run_eval(
                    self.task_config.context.env_config,
                    self.task_config.context.rl_config,
                    parameter,
                )
                if eval_rewards is not None:
                    for i, r in enumerate(eval_rewards):
                        d[f"eval_reward{i}"] = r

        self._base.write_log("client", d)
