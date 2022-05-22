import logging
from abc import abstractmethod
from typing import Any, Dict

import numpy as np
from srl.base.define import EnvObservationType, Info, RLActionType, RLObservationType
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker

logger = logging.getLogger(__name__)


class RuleBaseConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.ANY

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.ANY

    def _set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        pass  #


class RuleBaseParameter(RLParameter):
    def restore(self, data: Any) -> None:
        pass  # do nothing

    def backup(self):
        return None


class RuleBaseRemoteMemory(RLRemoteMemory):
    def length(self) -> int:
        return 0

    def restore(self, data: Any) -> None:
        pass  # do nothing

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
    def __init__(self):
        pass  # do nothing

    def _on_reset(self, *args) -> None:
        raise NotImplementedError()

    def _policy(self, *args) -> None:
        raise NotImplementedError()

    def _on_step(self, *args) -> None:
        raise NotImplementedError()

    # ----------------

    def call_on_reset(self, env: object) -> None:
        pass  # do nothing

    def on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.call_on_reset(env.get_original_env())

    @abstractmethod
    def call_policy(self, env: object) -> Any:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Any:
        return self.call_policy(env.get_original_env())

    def on_step(self, *args) -> Info:
        return {}

    def call_render(self, env: object) -> None:
        pass  # do nothing

    def render(self, env: EnvBase, player_index: int) -> None:
        self.call_render(env.get_original_env())


class GeneralWorker(RLWorker):
    def call_on_reset(self, env: object) -> None:
        pass  # do nothing

    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.call_on_reset(env.get_original_env())

    @abstractmethod
    def call_policy(self, env: EnvBase, player_index: int) -> Any:
        raise NotImplementedError()

    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Any:
        return self.call_policy(env, player_index)

    def _on_step(self, *args, **kwargs) -> Info:
        return {}

    def call_render(self, env: object) -> None:
        pass  # do nothing

    def render(self, env: EnvBase, player_index: int) -> None:
        self.call_render(env.get_original_env())
