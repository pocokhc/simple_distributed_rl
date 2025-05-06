import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from srl.base.context import RunContext, RunState
from srl.base.define import PlayerType
from srl.base.rl.parameter import RLParameter
from srl.base.run.core_play import RunStateActor, play
from srl.base.run.core_train_only import RunStateTrainer

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

    def create_eval_runner(self, context: RunContext):
        context = context.copy()
        context.run_name = "eval"
        context.seed = None  # mainと競合するのでNone
        context.flow_mode = "callback_evaluate"
        # stop config
        context.max_episodes = self.eval_episode
        context.timeout = self.eval_timeout
        context.max_steps = self.eval_max_steps
        context.max_train_count = 0
        context.max_memory = 0
        # play config
        context.players = self.eval_players
        context.shuffle_player = self.eval_shuffle_player
        context.disable_trainer = True
        # play info
        context.distributed = False
        context.training = False
        context.train_only = False
        context.rollout = False
        context.env_render_mode = ""
        context.rl_render_mode = ""

        state = RunState()
        state.memory = context.rl_config.make_memory()
        return (context, state)

    def create_eval_runner_if_not_exists(self, context: RunContext, state: Union[RunStateActor, RunStateTrainer]):
        if "eval_runner" not in state.shared_vars:
            state.shared_vars["eval_runner"] = self.create_eval_runner(context)
        return state.shared_vars["eval_runner"]

    def run_eval(self, context, state, parameter: RLParameter):
        if not self.enable_eval:
            return None
        try:
            state.parameter = parameter
            play(context, state)
            return np.mean(state.episode_rewards_list, axis=0)
        except Exception:
            logger.error(traceback.format_exc())
        return None

    def run_eval_state(self, context, state: Union[RunStateActor, RunStateTrainer]):
        if not self.enable_eval:
            return None
        try:
            c, s = self.create_eval_runner_if_not_exists(context, state)
            return self.run_eval(c, s, state.parameter)
        except Exception:
            logger.error(traceback.format_exc())
        return None
