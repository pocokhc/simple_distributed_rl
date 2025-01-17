import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from srl.base.context import RunContext
from srl.base.define import PlayerType
from srl.runner.runner_facade_distribution import RunnerFacadeDistribution
from srl.runner.runner_facade_play import RunnerFacadePlay
from srl.runner.runner_facade_train import RunnerFacadeTrain

logger = logging.getLogger(__name__)


@dataclass
class Runner(
    RunnerFacadeTrain,
    RunnerFacadePlay,
    RunnerFacadeDistribution,
):
    @staticmethod
    def create(context: RunContext):
        context = context.copy()
        return Runner(context.env_config, context.rl_config, context)

    def evaluate_compare_to_baseline_single_player(
        self,
        episode: int = -1,
        baseline: Optional[float] = None,
        eval_kwargs: dict = {},
        enable_backup_restore: bool = True,
    ):
        # baseline
        env = self.make_env()
        assert env.player_num == 1

        if episode <= 0:
            assert isinstance(env.reward_baseline, dict)
            episode = env.reward_baseline.get("episode", 0)
        if episode <= 0:
            episode = 100

        if baseline is None:
            assert isinstance(env.reward_baseline, dict)
            baseline = env.reward_baseline.get("baseline", None)
        assert baseline is not None, "Please specify a 'baseline'."

        # check restore
        if enable_backup_restore:
            parameter = self.make_parameter()
            parameter.restore(parameter.backup())

        # eval
        rewards = self.evaluate(max_episodes=episode, **eval_kwargs)

        # check
        reward = np.mean(rewards)
        result = reward >= baseline
        logger.info(f"{result}: {reward} >= {baseline}(baseline)")

        return result

    def evaluate_compare_to_baseline_multiplayer(
        self,
        players: List[PlayerType] = [],
        baseline_params: List[dict] = [],
        eval_kwargs: dict = {},
        enable_backup_restore: bool = True,
    ):
        # baseline
        env = self.make_env()
        assert env.player_num > 1

        if baseline_params == []:
            if env.reward_baseline is not None:
                baseline_params = env.reward_baseline
        assert isinstance(baseline_params, list)

        # check restore
        if enable_backup_restore:
            parameter = self.make_parameter()
            parameter.restore(parameter.backup())

        results = []
        for params in baseline_params:
            episode = params.get("episode", 100)
            players = params.get("players", [])
            baseline = params["baseline"]

            # eval
            rewards = self.evaluate(
                max_episodes=episode,
                players=players,
                **eval_kwargs,
            )

            # check
            rewards = np.mean(rewards, axis=0)
            result = []
            logger.info(f"baseline {baseline}, rewards {rewards}")
            for i, reward in enumerate(rewards):
                if baseline[i] is None:
                    result.append(True)
                else:
                    result.append(bool(reward >= baseline[i]))
            logger.info(f"{result=}")
            results.append(result)

        return results
