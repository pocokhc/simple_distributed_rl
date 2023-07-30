import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback, CallbackType, TrainerCallback
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(Callback, TrainerCallback):
    interval: int = 60 * 20  # s

    enable_eval: bool = True
    eval_env_sharing: bool = False
    eval_episode: int = 10
    eval_timeout: int = -1
    eval_max_steps: int = -1
    eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    eval_shuffle_player: bool = False
    eval_seed: Optional[int] = None
    eval_used_device_tf: str = "/CPU"
    eval_used_device_torch: str = "cpu"
    eval_callbacks: List[CallbackType] = field(default_factory=list)

    def __post_init__(self):
        self.env = None

    def _init(self, runner: Runner):
        self.save_dir = os.path.join(runner.context.save_dir, "checkpoints")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        self._create_eval_runner(runner)

    def _create_eval_runner(self, runner: Runner):
        if not self.enable_eval:
            return
        self.eval_runner = runner.create_eval_runner(self.eval_env_sharing)

        # config
        self.eval_runner.config.players = self.eval_players
        self.eval_runner.config.seed = self.eval_seed
        self.eval_runner.context.used_device_tf = self.eval_used_device_tf
        self.eval_runner.context.used_device_torch = self.eval_used_device_torch

        # context
        self.eval_runner.context.max_episodes = self.eval_episode
        self.eval_runner.context.timeout = self.eval_timeout
        self.eval_runner.context.max_steps = self.eval_max_steps
        self.eval_runner.context.shuffle_player = self.eval_shuffle_player
        self.eval_runner.context.callbacks = self.eval_callbacks

    def _eval(self, runner: Runner) -> str:
        if not self.enable_eval:
            return ""
        self.eval_runner._play(runner.parameter, runner.remote_memory)
        eval_rewards = self.eval_runner.state.episode_rewards_list
        eval_rewards = np.mean(eval_rewards, axis=0)
        return eval_rewards

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

    def on_trainer_train(self, runner: Runner):
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

        if self.enable_eval:
            eval_rewards = self._eval(runner)
            fn = f"{train_count}_{eval_rewards}.pickle"
        else:
            fn = f"{train_count}.pickle"

        runner.parameter.save(os.path.join(self.save_dir, fn))
