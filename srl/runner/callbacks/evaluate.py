import logging
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from srl.base.define import PlayRenderMode
from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.runner.play_sequence import play

logger = logging.getLogger(__name__)


@dataclass
class Evaluate(Callback):
    env_sharing: bool = True
    interval: int = 0  # episode
    num_episode: int = 1
    eval_players: List[Union[None, str, RLConfig]] = field(default_factory=list)

    def __post_init__(self):
        self.eval_config = None

    def _create_eval_config(self, config: Config):
        eval_config = config.copy(env_share=self.env_sharing)

        # random
        eval_config.seed = None

        eval_config.players = self.eval_players[:]
        eval_config.rl_config.remote_memory_path = ""

        # stop config
        eval_config.max_steps = -1
        eval_config.max_episodes = self.num_episode
        eval_config.timeout = -1
        # play config
        eval_config.shuffle_player = True
        eval_config.disable_trainer = True
        eval_config.render_mode = PlayRenderMode.none
        eval_config.enable_profiling = False
        # callbacks
        eval_config.callbacks = []

        # play info
        eval_config.training = False
        eval_config.distributed = False

        eval_config.run_name = "eval"
        return eval_config

    def on_episodes_begin(self, info) -> None:
        if info["actor_id"] != 0:
            return

        self.eval_config = self._create_eval_config(info["config"])
        self.eval_episode = 0

    def on_episode_end(self, info) -> None:
        if self.eval_config is None:
            return

        self.eval_episode += 1
        if self.eval_episode > self.interval:
            eval_rewards, _, _, _ = play(
                self.eval_config,
                parameter=info["parameter"],
            )
            info["eval_rewards"] = np.mean(eval_rewards, axis=0)
            self.eval_episode = 0

    # --- Trainer
    def on_trainer_start(self, info) -> None:
        config: Config = info["config"]
        if config.distributed:
            return

        self.eval_config = self._create_eval_config(config)

    def on_trainer_train(self, info) -> None:
        if self.eval_config is None:
            return

        train_count = info["train_count"]
        if train_count % (self.interval + 1) == 0:
            eval_rewards, _, _, _ = play(
                self.eval_config,
                parameter=info["parameter"],
            )
            info["eval_rewards"] = np.mean(eval_rewards, axis=0)
