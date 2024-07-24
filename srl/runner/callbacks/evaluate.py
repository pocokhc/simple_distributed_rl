import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from srl.base.context import RunContext, RunNameTypes
from srl.base.define import PlayerType
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
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

    def create_eval_runner_if_not_exists(
        self, context: RunContext, state: Union[RunStateActor, RunStateTrainer]
    ) -> Runner:
        if "eval_runner" in state.shared_vars:
            return state.shared_vars["eval_runner"]

        # --- make instance
        runner = Runner(context.env_config, context.rl_config)
        runner.context.run_name = RunNameTypes.eval
        runner.context.seed = None  # mainと競合するのでNone
        runner.make_memory(is_load=False)
        state.shared_vars["eval_runner"] = runner
        return runner

    def run_eval(self, context, state: Union[RunStateActor, RunStateTrainer]):
        if not self.enable_eval:
            return None

        try:
            runner = self.create_eval_runner_if_not_exists(context, state)
            runner.evaluate(
                self.eval_episode,
                self.eval_timeout,
                self.eval_max_steps,
                players=self.eval_players,
                shuffle_player=self.eval_shuffle_player,
                parameter=state.parameter,
                enable_progress=False,
            )
            eval_rewards = np.mean(runner.state.episode_rewards_list, axis=0)
            return eval_rewards
        except Exception:
            logger.error(traceback.format_exc())
        return None
