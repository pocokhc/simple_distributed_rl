import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from srl.runner.callback import Callback
from srl.runner.core import EvalOption

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(Callback):
    save_dir: str
    checkpoint_interval: int = 60 * 20  # s
    eval: Optional[EvalOption] = None

    def __post_init__(self):
        self.env = None
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"create dirs: {self.save_dir}")

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, info):
        self.t0 = time.time()
        if info["trainer"] is None:
            logger.info("checkpoint disable.")

    def on_episode_end(self, info):
        if info["trainer"] is None:
            return
        _time = time.time()
        if _time - self.t0 > self.checkpoint_interval:
            self.t0 = _time
            self._save_parameter(info["parameter"], info["trainer"].get_train_count(), info["config"])

    def on_episodes_end(self, info) -> None:
        if info["trainer"] is None:
            return
        self._save_parameter(info["parameter"], info["trainer"].get_train_count(), info["config"])

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        self.t0 = time.time()

    def on_trainer_train(self, info):
        _time = time.time()
        if _time - self.t0 > self.checkpoint_interval:
            self.t0 = _time
            self._save_parameter(info["parameter"], info["train_count"], info["config"])

    def on_trainer_end(self, info):
        self._save_parameter(info["parameter"], info["train_count"], info["config"])

    # ---------------------------
    # function
    # ---------------------------
    def _save_parameter(
        self,
        parameter,
        train_count,
        config,
    ):
        if train_count <= 0:
            logger.info("parameter save skip. (train count 0)")
            return
        if self.eval is not None:
            from srl.runner.callbacks.evaluate import Evaluate

            if isinstance(self.eval, dict):
                eval = Evaluate(**self.eval)
            else:
                eval = Evaluate(**asdict(self.eval))
            eval.create_eval_config(config)
            eval_rewards = eval.evaluate(parameter)

            fn = f"{train_count}_{eval_rewards}.pickle"
        else:
            fn = f"{train_count}.pickle"

        parameter.save(os.path.join(self.save_dir, fn))
