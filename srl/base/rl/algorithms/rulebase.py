import logging
from abc import abstractmethod
from typing import Any, Dict

from srl.base.define import EnvObservationType, Info, RLAction, RLActionType, RLObservation, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker

logger = logging.getLogger(__name__)


class RuleBaseConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.ANY

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.ANY

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def getName() -> str:
        return "RuleBaseConfig"

    def _set_config_by_env(
        self,
        env: EnvRun,
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
    def call_on_reset(self, env: object) -> None:
        pass  # do nothing

    @abstractmethod
    def call_policy(self, env: object) -> RLAction:
        raise NotImplementedError()

    # option
    def call_render(self, env: object) -> None:
        pass  # do nothing

    # ----------------

    def _call_on_reset(self, state: RLObservation, env: EnvRun) -> None:
        return self.call_on_reset(env.get_original_env())

    def _call_policy(self, status: RLObservation, env: EnvRun) -> RLAction:
        return self.call_policy(env.get_original_env())

    def _call_on_step(self, *args) -> None:
        pass  # do nothing

    def _call_render(self, env: EnvRun) -> None:
        self.call_render(env.get_original_env())


class GeneralWorker(RLWorker):
    def call_on_reset(self, env: object) -> None:
        pass  # do nothing

    def _call_on_reset(self, status: RLObservation, env: EnvRun) -> None:
        self.call_on_reset(env.get_original_env())

    @abstractmethod
    def call_policy(self, env: EnvRun, player_index: int) -> Any:
        raise NotImplementedError()

    def _call_policy(self, status: RLObservation, env: EnvRun) -> RLAction:
        return self.call_policy(env, self.player_index)

    def _call_on_step(self, *args, **kwargs) -> Info:
        return {}

    def _call_render(self, env: EnvRun) -> None:
        pass  # do nothing
