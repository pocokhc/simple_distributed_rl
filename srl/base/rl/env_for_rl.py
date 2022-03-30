import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union, cast

import gym
import gym.spaces
from srl.base.define import EnvObservationType, RLActionType
from srl.base.env.env import EnvBase, GymEnvWrapper
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.processor_observation_box import ObservationBoxProcessor

logger = logging.getLogger(__name__)


@dataclass
class EnvForRL(EnvBase):

    env: Union[gym.Env, EnvBase]
    rl_config: RLConfig

    override_env_observation_type: EnvObservationType = EnvObservationType.UNKOWN
    prediction_by_simulation: bool = True

    processors: List[Processor] = field(default_factory=list)

    # コンストラクタ
    def __post_init__(self):
        self._valid_actions = None  # cache

        # gym env
        if not issubclass(self.env.unwrapped.__class__, EnvBase):
            self.env = GymEnvWrapper(self.env)
        self.env = cast(EnvBase, self.env)

        # env info
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        observation_type = self.env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKOWN:
            observation_type = self.override_env_observation_type

        # ObsはBox/np.ndarrayで統一する(processorsの先頭に入れる)
        self.processors.insert(0, ObservationBoxProcessor(self.env, self.prediction_by_simulation))

        # processor
        for processor in self.processors:
            action_space = processor.change_action_info(
                action_space,
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
        self.after_observation_space = observation_space
        self.after_observation_type = observation_type
        logger.info(f"before_action          : {self._space_str(self.env.action_space)}")
        logger.info(f"before_observation     : {self._space_str(self.env.observation_space)}")
        logger.info(f"before_observation type: {self.env.observation_type}")
        logger.info(f"after_action           : {self._space_str(self.after_action_space)}")
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

    # ------------------------------------------
    # ABC method
    # ------------------------------------------

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.after_action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.after_observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self.after_observation_type

    @property
    def max_episode_steps(self) -> int:
        return self.env.max_episode_steps

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Any:
        state = self.env.reset()
        for processor in self.processors:
            state = processor.observation_encode(state)
        self._valid_actions = None
        return state

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        for processor in self.processors:
            action = processor.action_decode(action)
        state, reward, done, info = self.env.step(action)
        for processor in self.processors:
            state = processor.observation_encode(state)
        self._valid_actions = None
        return state, float(reward), bool(done), info

    def fetch_valid_actions(self) -> Optional[List[int]]:
        if self.rl_config.action_type == RLActionType.CONTINUOUS:
            return None

        if self._valid_actions is None:
            self._valid_actions = self.env.fetch_valid_actions()
            if self._valid_actions is None:
                for processor in self.processors:
                    self._valid_actions = processor.valid_actions_encode(self._valid_actions)

        return self._valid_actions

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def action_to_str(self, action: Any) -> str:
        return self.env.action_to_str(action)

    def backup(self) -> Any:
        return self.env.backup()

    def restore(self, state: Any) -> None:
        return self.env.restore(state)


if __name__ == "__main__":
    pass
