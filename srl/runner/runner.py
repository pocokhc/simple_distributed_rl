from dataclasses import dataclass

from srl.base.context import RunContext
from srl.runner.runner_facade_distribution import RunnerFacadeDistribution
from srl.runner.runner_facade_play import RunnerFacadePlay
from srl.runner.runner_facade_train import RunnerFacadeTrain


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
