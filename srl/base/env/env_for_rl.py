import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType
from srl.base.env import EnvBase
from srl.base.rl.base import RLConfig
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class EnvForRL(EnvBase):

    env: EnvBase
    rl_config: RLConfig

    override_env_observation_type: EnvObservationType = EnvObservationType.UNKOWN
    prediction_by_simulation: bool = True

    processors: List[Processor] = field(default_factory=list)

    # コンストラクタ
    def __post_init__(self):
        self._invalid_actions = None  # cache

        # env info
        action_space = self.env.action_space
        action_type = self.env.action_type
        observation_space = self.env.observation_space
        observation_type = self.env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKOWN:
            observation_type = self.override_env_observation_type

        # processor
        for processor in self.processors:
            action_space, action_type = processor.change_action_info(
                action_space,
                action_type,
                self.rl_config.action_type,
            )
            observation_space, observation_type = processor.change_observation_info(
                observation_space,
                observation_type,
                self.rl_config.observation_type,
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
            state = processor.observation_encode(state)
        return state

    def action_decode(self, action):
        for processor in self.processors:
            action = processor.action_decode(action)
        return action

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

    def reset(self) -> Tuple[List[np.ndarray], List[int]]:
        states, player_indexes = self.env.reset()
        states = [self.observation_encode(s) for s in states]
        self._invalid_actions = None
        return states, player_indexes

    def step(
        self, actions: List[Any], player_indexes: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[int], bool, dict]:
        actions = [self.action_decode(a) for a in actions]
        next_states, rewards, next_player_indexes, done, env_info = self.env.step(actions, player_indexes)
        self._invalid_actions = None
        next_states = [self.observation_encode(s) for s in next_states]
        rewards = [float(r) for r in rewards]
        return next_states, rewards, next_player_indexes, bool(done), env_info

    def fetch_invalid_actions(self) -> List[List[int]]:
        if self.rl_config.action_type == RLActionType.CONTINUOUS:
            return [None for _ in range(self.player_num)]

        if self._invalid_actions is None:
            self._invalid_actions = self.env.fetch_invalid_actions()
            for processor in self.processors:
                self._invalid_actions = [processor.invalid_actions_encode(va) for va in self._invalid_actions]

        return self._invalid_actions

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
