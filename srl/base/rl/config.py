from abc import ABC, abstractmethod
from typing import cast

from srl.base.define import EnvObservationType, RLActionType, RLObservationType


class RLConfig(ABC):
    def __init__(
        self,
        batch_size: int,
        memory_warmup_size: int,
    ):
        self.batch_size = batch_size
        self.memory_warmup_size = memory_warmup_size

    def assert_params(self) -> None:
        assert self.batch_size > 0
        assert (
            self.batch_size < self.memory_warmup_size
        ), f"batch_size={self.batch_size}, warmup_size={self.memory_warmup_size}"

    @staticmethod
    @abstractmethod
    def getName() -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_type(self) -> RLActionType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> RLObservationType:
        raise NotImplementedError()

    @abstractmethod
    def set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        raise NotImplementedError()


class TableConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    def set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self._nb_actions = env.action_space.n  # type: ignore

    @property
    def nb_actions(self) -> int:
        return self._nb_actions


class DiscreteActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self._nb_actions = env.action_space.n  # type: ignore
        self._env_observation_shape = cast(tuple, env.observation_space.shape)  # type: ignore
        self._env_observation_type = env.observation_type

    @property
    def nb_actions(self) -> int:
        return self._nb_actions

    @property
    def env_observation_shape(self) -> tuple:
        return self._env_observation_shape

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type


class ContinuousActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.CONTINUOUS

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        assert len(env.action_space.shape) == 1  # type: ignore
        self._action_num = env.action_space.shape[0]  # type: ignore
        self._action_low = env.action_space.low  # type: ignore
        self._action_high = env.action_space.high  # type: ignore
        self._env_observation_shape = cast(tuple, env.observation_space.shape)  # type: ignore
        self._env_observation_type = env.observation_type

    @property
    def action_num(self) -> int:
        return self._action_num

    @property
    def action_low(self) -> tuple:
        return self._action_low

    @property
    def action_high(self) -> tuple:
        return self._action_high

    @property
    def env_observation_shape(self) -> tuple:
        return self._env_observation_shape

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type


if __name__ == "__main__":
    pass
