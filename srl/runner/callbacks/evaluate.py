import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from srl.base.rl.base import RLParameter
from srl.base.run.context import RLWorkerType, StrWorkerType
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
    eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = field(default_factory=list)
    eval_shuffle_player: bool = False
    eval_enable_tf_device: bool = True
    eval_used_device_tf: str = "/CPU"
    eval_used_device_torch: str = "cpu"
    eval_callbacks: List[CallbackType] = field(default_factory=list)

    def setup_eval_runner(self, runner: Runner):
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

    def run_eval(self, parameter: Optional[RLParameter]) -> np.ndarray:
        assert parameter is not None
        eval_rewards = self.eval_runner.callback_play_eval(parameter)
        eval_rewards = np.mean(eval_rewards, axis=0)
        return eval_rewards
