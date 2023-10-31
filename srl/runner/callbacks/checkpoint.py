import datetime
import glob
import logging
import os
import time
import traceback
from dataclasses import dataclass

from srl.runner.callback import Callback, TrainerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(Callback, TrainerCallback, Evaluate):
    save_dir: str = "checkpoints"
    interval: int = 60 * 20  # s

    def _init(self, runner: Runner):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        self.setup_eval_runner(runner)

    @staticmethod
    def get_parameter_path(save_dir: str) -> str:
        # 最後のpathを取得
        trains = []
        for f in glob.glob(os.path.join(save_dir, "*.pickle")):
            try:
                date = os.path.basename(f).split("_")[0]
                date = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
                trains.append([date, f])
            except Exception:
                logger.warning(traceback.format_exc())
        if len(trains) == 0:
            return ""
        trains.sort()
        return trains[-1][1]

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, runner: Runner):
        self._init(runner)
        self.interval_t0 = time.time()
        self._save_parameter(runner, is_last=False)

    def on_episode_end(self, runner: Runner):
        if runner.state.trainer is None:
            return
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(runner, is_last=False)
            self.interval_t0 = time.time()

    def on_episodes_end(self, runner: Runner) -> None:
        if runner.state.trainer is None:
            return
        self._save_parameter(runner, is_last=True)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, runner: Runner):
        self._init(runner)
        self.interval_t0 = time.time()
        self._save_parameter(runner, is_last=False)

    def on_trainer_loop(self, runner: Runner):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(runner, is_last=False)
            self.interval_t0 = time.time()

    def on_trainer_end(self, runner: Runner):
        self._save_parameter(runner, is_last=True)

    # ---------------------------
    # function
    # ---------------------------
    def _save_parameter(self, runner: Runner, is_last: bool):
        if runner.state.trainer is None:
            return
        train_count = runner.state.trainer.get_train_count()
        assert runner.state.parameter is not None

        if self.enable_eval:
            eval_rewards = self.run_eval(runner.state.parameter)
        else:
            eval_rewards = "None"

        fn = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn += f"_{train_count}_{eval_rewards}"
        if is_last:
            fn += "_last"
        fn += ".pickle"

        runner.state.parameter.save(os.path.join(self.save_dir, fn))
