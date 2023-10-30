import logging
import os
import time
from dataclasses import dataclass

from srl.runner.callback import Callback, TrainerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(Callback, TrainerCallback, Evaluate):
    interval: int = 60 * 20  # s

    def __post_init__(self):
        self.env = None

    def _init(self, runner: Runner):
        self.save_dir = os.path.join(runner.context.wkdir, "checkpoints")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        self.setup_eval_runner(runner)

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, runner: Runner):
        self._init(runner)

        self.interval_t0 = time.time()
        if runner.state.trainer is None:
            logger.info("checkpoint disable.")

    def on_episode_end(self, runner: Runner):
        if runner.state.trainer is None:
            return
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(runner)
            self.interval_t0 = time.time()

    def on_episodes_end(self, runner: Runner) -> None:
        if runner.state.trainer is None:
            return
        self._save_parameter(runner)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, runner: Runner):
        self._init(runner)

        self.interval_t0 = time.time()

    def on_trainer_loop(self, runner: Runner):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(runner)
            self.interval_t0 = time.time()

    def on_trainer_end(self, runner: Runner):
        self._save_parameter(runner)

    # ---------------------------
    # function
    # ---------------------------
    def _save_parameter(self, runner: Runner):
        if runner.state.trainer is None:
            return
        train_count = runner.state.trainer.get_train_count()
        if train_count <= 0:
            logger.info("parameter save skip. (train count 0)")
            return
        assert runner.state.parameter is not None

        if self.enable_eval:
            eval_rewards = self.run_eval(runner.state.parameter)
            fn = f"{train_count}_{eval_rewards}.pickle"
        else:
            fn = f"{train_count}.pickle"

        runner.state.parameter.save(os.path.join(self.save_dir, fn))
