import logging
from typing import Any, Dict, List, Tuple, Union, cast

from srl.base.rl.algorithms.rulebase import (
    RuleBaseConfig,
    RuleBaseParamete,
    RuleBaseRemoteMemory,
    RuleBaseTrainer,
    RuleBaseWorker,
)
from srl.base.rl.base import RLWorker
from srl.base.rl.registration import register

logger = logging.getLogger(__name__)


class Config(RuleBaseConfig):
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


class Parameter(RuleBaseParamete):
    pass


class RemoteMemory(RuleBaseRemoteMemory):
    pass


class Trainer(RuleBaseTrainer):
    pass


class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.action_num = self.config.env_action_space.n

    def on_reset(self, *args, **kwargs) -> None:
        pass

    def policy(self, state, invalid_actions, *args, **kwargs) -> Any:
        actions = [a for a in range(self.action_num) if a not in invalid_actions]
        print(f"select action: {actions}")
        for _ in range(10):
            action = int(input("> "))
            if action in actions:
                break
            print("out of range")

        return action

    def on_step(self, *args, **kwargs):
        return {}

    def render(self, *args, **kwargs):
        pass
