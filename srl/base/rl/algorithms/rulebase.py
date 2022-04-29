import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from srl.base.define import RLActionType, RLObservationType
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker

logger = logging.getLogger(__name__)


class RuleBaseConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.ANY

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.ANY

    def _set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self.env_observation_type = env.observation_type
        self.env_action_space = env.action_space


class RuleBaseParamete(RLParameter):
    def restore(self, data: Any) -> None:
        pass

    def backup(self):
        return None


class RuleBaseRemoteMemory(RLRemoteMemory):
    def length(self) -> int:
        return 0

    def restore(self, data: Any) -> None:
        pass

    def backup(self):
        return None


class RuleBaseTrainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.train_count = 0

    def get_train_count(self) -> int:
        return self.train_count

    def train(self) -> Dict[str, Any]:
        self.train_count += 1
        return {}


class RuleBaseWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, env: object) -> None:
        raise NotImplementedError()

    def on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> None:
        self.call_on_reset(env.get_original_env())

    @abstractmethod
    def call_policy(self, env: object) -> Any:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> Any:
        return self.call_policy(env.get_original_env())

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> Dict[str, Union[float, int]]:  # info
        return {}

    def call_render(self, env: object) -> None:
        pass

    def render(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self.call_render(env.get_original_env())


if __name__ == "__main__":
    pass
