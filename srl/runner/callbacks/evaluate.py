import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from srl.base.rl.parameter import RLParameter
from srl.base.run.context import RLWorkerType, StrWorkerType
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


# envを共有すると状態が変わるのでやめる
@dataclass
class Evaluate:
    enable_eval: bool = True
    eval_episode: int = 1
    eval_timeout: float = -1
    eval_max_steps: int = -1
    eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = field(default_factory=list)
    eval_shuffle_player: bool = False

    _eval_runner: Optional[Runner] = field(init=False, default=None)

    def setup_eval_runner(self, runner: Runner) -> bool:
        if not self.enable_eval:
            return False
        if self._eval_runner is not None:
            return True
        self._eval_runner = runner.create_eval_runner(
            self.eval_episode,
            self.eval_timeout,
            self.eval_max_steps,
            self.eval_players,
            self.eval_shuffle_player,
        )
        return True

    def run_eval(self, parameter: RLParameter) -> Optional[np.ndarray]:
        if self._eval_runner is None:
            return None
        try:
            eval_rewards = self._eval_runner.callback_play_eval(parameter)
            eval_rewards = np.mean(eval_rewards, axis=0)
            return eval_rewards
        except Exception:
            logger.error(traceback.format_exc())
            return None
