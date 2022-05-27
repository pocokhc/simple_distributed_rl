import logging
from typing import Any, cast

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
        return "Human"


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

        # Con TODO
        self.action_num = self.config.env_action_space.n

    def call_policy(self, env: EnvRun, player_index: int) -> Any:
        invalid_actions = env.get_invalid_actions(player_index)
        actions = [a for a in range(self.action_num) if a not in invalid_actions]
        print(f"select action: {actions}")
        for _ in range(10):
            try:
                action = int(input("> "))
                if action in actions:
                    break
            except Exception:
                print("invalid action")

        return action
