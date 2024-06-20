import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, cast

import numpy as np

from srl.base.context import RunContext, RunNameTypes
from srl.base.define import PlayerType
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.run import play
from srl.base.run.core_play import RunStateActor

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

    _context: Optional[RunContext] = field(init=False, default=None)

    def run_eval(self, env_config: EnvConfig, rl_config: RLConfig, parameter: RLParameter) -> Optional[np.ndarray]:
        if not self.enable_eval:
            return None
        if self._context is None:
            context = RunContext()
            context.players = self.eval_players
            context.run_name = RunNameTypes.eval
            # stop
            context.max_episodes = self.eval_episode
            context.timeout = self.eval_timeout
            context.max_steps = self.eval_max_steps
            context.max_train_count = -1
            context.max_memory = -1
            # play config
            context.shuffle_player = self.eval_shuffle_player
            context.disable_trainer = True
            # play info
            context.distributed = False
            context.training = False
            context.rendering = False
            context.seed = None  # mainと競合するのでNone
            # thread
            context.use_train_thread = False
            self._context = context

            # --- make instance
            self._env = env_config.make()
            self._memory = rl_config.make_memory(is_load=False)
            self._workers, self._main_worker_idx = rl_config.make_workers(
                context.players, self._env, parameter, self._memory
            )

        if self._context is None:
            return None
        try:
            state = cast(
                RunStateActor,
                play.play(self._context, self._env, self._workers, self._main_worker_idx),
            )
            eval_rewards = np.mean(state.episode_rewards_list, axis=0)
            return eval_rewards
        except Exception:
            logger.error(traceback.format_exc())
            return None
