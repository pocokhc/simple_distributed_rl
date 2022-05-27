import logging

from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.rulebase import (
    GeneralWorker,
    RuleBaseConfig,
    RuleBaseParameter,
    RuleBaseRemoteMemory,
    RuleBaseTrainer,
)
from srl.base.rl.registration import register

logger = logging.getLogger(__name__)


class Config(RuleBaseConfig):
    def __init__(self):
        super().__init__()

    @staticmethod
    def getName() -> str:
        return "Random"


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Parameter(RuleBaseParameter):
    pass


class RemoteMemory(RuleBaseRemoteMemory):
    pass


class Trainer(RuleBaseTrainer):
    pass


class Worker(GeneralWorker):
    def call_policy(self, env: EnvRun, player_index: int):
        invalid_actions = self.get_invalid_actions(env)
        return self.config.env_action_space.sample(invalid_actions)
