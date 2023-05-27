import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from srl import runner
from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback
from srl.runner.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Evaluate(Callback):
    env_sharing: bool = True
    interval: int = 0  # episode
    # stop config
    episode: int = 1
    timeout: int = -1
    max_steps: int = -1
    # play config
    players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    shuffle_player: bool = False
    # tensorflow options
    device: str = "CPU"
    tf_disable: bool = True
    # random
    seed: Optional[int] = None
    # other
    callbacks: List[Callback] = field(default_factory=list)

    def __post_init__(self):
        self.eval_config = None

    def create_eval_config(self, config: Config):
        eval_config = config.copy(env_share=self.env_sharing)

        eval_config.rl_config.remote_memory_path = ""

        eval_config.seed = self.seed
        eval_config.device_main = self.device
        eval_config.tf_disable = self.tf_disable
        eval_config.players = self.players

        self.eval_config = eval_config

    def evaluate(self, parameter):
        if self.eval_config is None:
            return
        eval_rewards = runner.evaluate(
            self.eval_config,
            parameter=parameter,
            max_episodes=self.episode,
            timeout=self.timeout,
            max_steps=self.max_steps,
            shuffle_player=self.shuffle_player,
            progress=None,
            callbacks=self.callbacks,
        )
        if self.eval_config.env_config.player_num == 1:
            eval_rewards = [np.mean(eval_rewards)]
        else:
            eval_rewards = np.mean(eval_rewards, axis=0)
        return eval_rewards

    # --- Actor
    def on_episodes_begin(self, info) -> None:
        if info["config"].actor_id != 0:
            return

        self.create_eval_config(info["config"])
        self.eval_episode = 0

    def on_episode_end(self, info) -> None:
        if self.eval_config is None:
            return

        self.eval_episode += 1
        if self.eval_episode > self.interval:
            info["eval_rewards"] = self.evaluate(info["parameter"])
            self.eval_episode = 0

    # --- Trainer
    def on_trainer_start(self, info) -> None:
        config: Config = info["config"]
        if config.distributed:
            return

        self.create_eval_config(config)

    def on_trainer_train(self, info) -> None:
        if self.eval_config is None:
            return

        train_count = info["train_count"]
        if train_count % (self.interval + 1) == 0:
            info["eval_rewards"] = self.evaluate(info["parameter"])
