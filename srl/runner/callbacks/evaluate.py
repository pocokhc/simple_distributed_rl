import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from srl.base.rl.config import RLConfig
from srl.runner.callback import CallbackType
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class Evaluate:
    enable_eval: bool = True
    eval_env_sharing: bool = False
    eval_episode: int = 1
    eval_timeout: int = -1
    eval_max_steps: int = -1
    eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    eval_shuffle_player: bool = False
    eval_enable_tf_device: bool = True
    eval_used_device_tf: str = "/CPU"
    eval_used_device_torch: str = "cpu"
    eval_callbacks: List[CallbackType] = field(default_factory=list)

    def create_eval_runner(self, runner: Runner):
        if not self.enable_eval:
            return
        self.eval_runner = runner.create_eval_runner(
            self.eval_env_sharing,
            self.eval_episode,
            self.eval_timeout,
            self.eval_max_steps,
            self.eval_players,
            self.eval_shuffle_player,
            self.eval_enable_tf_device,
            self.eval_used_device_tf,
            self.eval_used_device_torch,
            self.eval_callbacks,
        )

    def run_eval(self, runner: Runner) -> Optional[np.ndarray]:
        if not self.enable_eval:
            return None
        assert runner.parameter is not None
        eval_rewards = self.eval_runner.callback_play_eval(runner.parameter)
        eval_rewards = np.mean(eval_rewards, axis=0)
        return eval_rewards
