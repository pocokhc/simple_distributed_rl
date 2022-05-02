import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType
from srl.base.env import EnvBase
from srl.base.env.processor import Processor
from srl.base.env.processors import ContinuousProcessor, DiscreteProcessor, ObservationBoxProcessor
from srl.base.rl.base import RLConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    name: str
    kwargs: Dict = None

    processors: List[Processor] = None
    override_env_observation_type: EnvObservationType = EnvObservationType.UNKOWN
    prediction_by_simulation: bool = True
    action_division_num: int = 5
    observation_division_num: int = 50

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.processors is None:
            self.processors = []


@dataclass
class EnvForRL(EnvBase):

    env: EnvBase
    rl_config: RLConfig
    env_config: EnvConfig

    # コンストラクタ
    def __post_init__(self):

        # processors
        base_processors = [
            ObservationBoxProcessor(self.env_config.prediction_by_simulation, self.env_config),
            DiscreteProcessor(self.env_config.action_division_num, self.env_config.observation_division_num),
            ContinuousProcessor(),
        ]
        self.processors = self.env_config.processors + base_processors

        # env info
        action_space = self.env.action_space
        action_type = self.env.action_type
        observation_space = self.env.observation_space
        observation_type = self.env.observation_type

        # observation_typeの上書き
        if self.env_config.override_env_observation_type != EnvObservationType.UNKOWN:
            observation_type = self.env_config.override_env_observation_type

        # processor
        for processor in self.processors:
            action_space, action_type = processor.change_action_info(
                action_space,
                action_type,
                self.rl_config.action_type,
                self.env.get_original_env(),
            )
            observation_space, observation_type = processor.change_observation_info(
                observation_space,
                observation_type,
                self.rl_config.observation_type,
                self.env.get_original_env(),
            )

        # unknownはcontinuousとする
        if observation_type == EnvObservationType.UNKOWN:
            observation_type = EnvObservationType.CONTINUOUS

        # 変更後
        self.after_action_space = action_space
        self.after_action_type = action_type
        self.after_observation_space = observation_space
        self.after_observation_type = observation_type
        logger.info(f"before_action          : {self._space_str(self.env.action_space)}")
        logger.info(f"before_action type     : {self.action_type}")
        logger.info(f"before_observation     : {self._space_str(self.env.observation_space)}")
        logger.info(f"before_observation type: {self.env.observation_type}")
        logger.info(f"after_action           : {self._space_str(self.after_action_space)}")
        logger.info(f"after_action type      : {self.after_action_type}")
        logger.info(f"after_observation      : {self._space_str(self.after_observation_space)}")
        logger.info(f"after_observation type : {self.after_observation_type}")
        logger.info(f"rl_action type      : {self.rl_config.action_type}")
        logger.info(f"rl_observation type : {self.rl_config.observation_type}")

        # RLConfig側を設定する
        self.rl_config.set_config_by_env(self)

    def _space_str(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return f"{space.__class__.__name__} {space.n}"
        if isinstance(space, gym.spaces.Tuple):
            return f"{space.__class__.__name__} {len(space)}"
        return f"{space.__class__.__name__}{space.shape} ({space.low.flatten()[0]} - {space.high.flatten()[0]})"

    def observation_encode(self, state):
        for processor in self.processors:
            state = processor.observation_encode(state, self.env.get_original_env())
        return state

    def action_decode(self, action):
        for processor in self.processors:
            action = processor.action_decode(action, self.env.get_original_env())
        return action

    @property
    def config(self) -> EnvConfig:
        return self.env_config

    # ------------------------------------------
    # ABC method
    # ------------------------------------------

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.after_action_space

    @property
    def action_type(self) -> EnvActionType:
        return self.after_action_type

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.after_observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self.after_observation_type

    @property
    def max_episode_steps(self) -> int:
        return self.env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.env.player_num

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state, player_indices = self.env.reset()
        state = self.observation_encode(state)
        return state, player_indices

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
        actions = [self.action_decode(a) for a in actions]
        next_state, rewards, done, next_player_indices, env_info = self.env.step(actions)
        next_state = self.observation_encode(next_state)
        rewards = [float(r) for r in rewards]
        return next_state, rewards, bool(done), next_player_indices, env_info

    def get_next_player_indecies(self) -> List[int]:
        return self.env.get_next_player_indecies()

    def get_invalid_actions(self, player_index: int) -> List[int]:
        if self.rl_config.action_type == RLActionType.CONTINUOUS:
            return []

        invalid_actions = self.env.get_invalid_actions(player_index)
        invalid_actions = [self._invalid_actions_encode(va) for va in invalid_actions]
        return invalid_actions

    def _invalid_actions_encode(self, action) -> int:
        for processor in self.processors:
            action = processor.invalid_actions_encode(action, self.env.get_original_env())
        return action

    def render(self, *args):
        return self.env.render(*args)

    def backup(self) -> Any:
        return self.env.backup()

    def restore(self, state: Any) -> None:
        return self.env.restore(state)

    def action_to_str(self, action: Any) -> str:
        return self.env.action_to_str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return self.env.make_worker(name)

    def get_original_env(self) -> object:
        return self.env.get_original_env()


if __name__ == "__main__":
    pass
