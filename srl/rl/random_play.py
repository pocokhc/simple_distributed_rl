import logging
from typing import cast

from srl.base.env.base import EnvBase
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
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

    def call_policy(self, env: EnvBase, player_index: int):
        invalid_actions = env.get_invalid_actions(player_index)
        return self.config.env_action_space.sample(invalid_actions)
