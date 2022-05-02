import logging
import random
from typing import Any, Dict, List, Union, cast

from srl.base.rl.algorithms.rulebase import RuleBaseConfig, RuleBaseParamete, RuleBaseRemoteMemory, RuleBaseTrainer
from srl.base.rl.base import RLWorker
from srl.base.rl.registration import register

logger = logging.getLogger(__name__)


class Config(RuleBaseConfig):
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

    def policy(self, state, player_index, env, *args, **kwargs):
        invalid_actions = env.get_invalid_actions(player_index)
        return random.choice([a for a in range(self.config.env_action_space.n) if a not in invalid_actions])

    def on_step(self, *args, **kwargs) -> Dict[str, Union[float, int]]:
        return {}

    def render(self, *args, **kwargs):
        pass
