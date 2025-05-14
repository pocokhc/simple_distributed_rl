import logging
import traceback
from dataclasses import dataclass, field

import numpy as np

from srl.base.context import RunContext, RunState
from srl.base.define import PlayersType
from srl.base.rl.parameter import RLParameter

logger = logging.getLogger(__name__)


# envを共有すると状態が変わるのでやめる
@dataclass
class Evaluate:
    enable_eval: bool = True
    eval_episode: int = 1
    eval_timeout: float = -1
    eval_max_steps: int = -1
    eval_players: PlayersType = field(default_factory=list)
    eval_shuffle_player: bool = False

    def create_eval_runner(self, context: RunContext):
        from srl.runner.runner import Runner

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
        return Runner.create(context)

    def create_eval_runner_if_not_exists(self, context: RunContext, state: RunState):
        if "eval_runner" not in state.shared_vars:
            runner = self.create_eval_runner(context)
            runner.state.parameter = state.parameter
            state.shared_vars["eval_runner"] = runner
        return state.shared_vars["eval_runner"]

    def run_eval_with_state(self, context: RunContext, state: RunState):
        if not self.enable_eval:
            return None
        try:
            runner = self.create_eval_runner_if_not_exists(context, state)
            runner.core_play()
            return np.mean(runner.state.episode_rewards_list, axis=0)
        except Exception:
            logger.error(traceback.format_exc())
        return None

    def run_eval(self, runner, parameter: RLParameter):
        if not self.enable_eval:
            return None
        try:
            runner.state.parameter = parameter
            runner.core_play()
            return np.mean(runner.state.episode_rewards_list, axis=0)
        except Exception:
            logger.error(traceback.format_exc())
        return None
