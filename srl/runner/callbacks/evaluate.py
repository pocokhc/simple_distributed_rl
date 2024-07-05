import logging
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from srl.base.context import RunNameTypes
from srl.base.define import PlayerType
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter

if TYPE_CHECKING:
    from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


# envを共有すると状態が変わるのでやめる
@dataclass
class Evaluate:
    enable_eval: bool = True
    eval_episode: int = 1
    eval_timeout: float = -1
    eval_max_steps: int = -1
    eval_players: List[PlayerType] = field(default_factory=list)
    eval_shuffle_player: bool = False

    _eval_runner: Optional["Runner"] = field(init=False, default=None)

    def run_eval(self, env_config: EnvConfig, rl_config: RLConfig, parameter: RLParameter) -> Optional[np.ndarray]:
        if not self.enable_eval:
            return None
        if self._eval_runner is None:
            from srl.runner.runner import Runner

            self._eval_runner = Runner(env_config, rl_config)
            c = self._eval_runner.context
            c.players = self.eval_players
            c.run_name = RunNameTypes.eval
            c.flow_mode = "Evaluate"
            # stop
            c.max_episodes = self.eval_episode
            c.timeout = self.eval_timeout
            c.max_steps = self.eval_max_steps
            c.max_train_count = -1
            c.max_memory = -1
            # play config
            c.shuffle_player = self.eval_shuffle_player
            c.disable_trainer = True
            # play info
            c.distributed = False
            c.training = False
            c.rendering = False
            c.seed = None  # mainと競合するのでNone
            # thread
            c.enable_train_thread = False

            # --- make instance
            self._eval_runner.make_memory(is_load=False)

        try:
            state = self._eval_runner.run_context(
                reset_workers=False,
                reset_trainer=False,
                parameter=parameter,
            )
            eval_rewards = np.mean(state.episode_rewards_list, axis=0)
            return eval_rewards
        except Exception:
            logger.error(traceback.format_exc())
            return None
