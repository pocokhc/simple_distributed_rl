import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from srl.base.define import RLActionType, RLObservationType
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class TableConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    def set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self._nb_actions = env.action_space.n
        self._is_set_config_by_env = True

    @property
    def nb_actions(self) -> int:
        return self._nb_actions


class TableWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        raise NotImplementedError()

    def on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
        start_player_indexes: List[int],
    ) -> None:
        self.call_on_reset(state, invalid_actions)

    @abstractmethod
    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, Any]:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
        player_indexes: List[int],
    ) -> Tuple[Any, Any]:
        return self.call_policy(state, invalid_actions)

    @abstractmethod
    def call_on_step(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        invalid_actions: List[int],
        next_invalid_actions: List[int],
    ) -> Dict[str, Union[float, int]]:  # info
        raise NotImplementedError()

    def on_step(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
        done: bool,
        invalid_actions: List[int],
        next_invalid_actions: List[int],
        env,
    ) -> Dict[str, Union[float, int]]:  # info
        return self.call_on_step(state, action, next_state, reward, done, invalid_actions, next_invalid_actions)


if __name__ == "__main__":
    pass
