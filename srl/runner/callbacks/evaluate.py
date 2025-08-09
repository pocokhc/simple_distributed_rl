import logging
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from srl.base.context import RunContext
from srl.base.define import PlayersType
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
    eval_players: PlayersType = field(default_factory=list)
    eval_shuffle_player: bool = False

    def create_eval_runner(self, context: RunContext):
        from srl.runner.runner import Runner

        context = context.copy(include_callbacks=False)
        context.run_name = "eval"
        context.seed = None  # mainと競合するのでNone
        context.max_episodes = self.eval_episode
        context.timeout = self.eval_timeout
        context.max_steps = self.eval_max_steps
        context.players = self.eval_players
        context.shuffle_player = self.eval_shuffle_player

        runner = Runner(context=context)
        runner.update_context_evaluate(enable_progress=False)
        return runner

    def run_eval_with_state(self, context: RunContext, state):
        if not self.enable_eval:
            return None
        try:
            if "eval_runner" not in state.shared_vars:
                state.shared_vars["eval_runner"] = self.create_eval_runner(context)
            runner: "Runner" = state.shared_vars["eval_runner"]

            # eval
            runner.parameter.restore(state.parameter.backup())
            state = runner.play_direct()
            return np.mean(state.episode_rewards_list, axis=0)
        except Exception:
            logger.error(traceback.format_exc())
        return None

    def run_eval(self, runner: "Runner", parameter: RLParameter):
        if not self.enable_eval:
            return None
        try:
            runner.parameter.restore(parameter.backup())
            state = runner.play_direct()
            return np.mean(state.episode_rewards_list, axis=0)
        except Exception:
            logger.error(traceback.format_exc())
        return None
