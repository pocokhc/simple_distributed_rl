import logging
import random
from dataclasses import dataclass

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.env import EnvBase
from srl.base.rl.processor import Processor
from srl.rl.processor.common import tuple_to_box

logger = logging.getLogger(__name__)


@dataclass
class ObservationBoxProcessor(Processor):
    """change Box & pred type"""

    env: EnvBase
    prediction_by_simulation: bool = True

    def __post_init__(self):
        self.change_type = ""

    def change_observation_info(
        self,
        observation_space: gym.spaces.Space,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
    ):
        if isinstance(observation_space, gym.spaces.Discrete):
            next_space = gym.spaces.Box(low=0, high=observation_space.n - 1, shape=(1,))
            self.change_type = "Discrete->Box"
            if observation_type == EnvObservationType.UNKOWN:
                observation_type = EnvObservationType.DISCRETE
            return next_space, observation_type

        if isinstance(observation_space, gym.spaces.Tuple):
            new_observation_space = tuple_to_box(observation_space)
            self.change_type = "Tuple->Box"

            # 予測
            if observation_type == EnvObservationType.UNKOWN:
                # すべてDiscreteならdiscrete
                _is_all_disc = True
                for s in observation_space.spaces:
                    if not isinstance(s, gym.spaces.Discrete):
                        _is_all_disc = False
                        break
                if _is_all_disc:
                    observation_type = EnvObservationType.DISCRETE
                else:
                    observation_type = self._pred_type_from_box(new_observation_space)

            return new_observation_space, observation_type

        if isinstance(observation_space, gym.spaces.Box):
            # 予測
            if observation_type == EnvObservationType.UNKOWN:
                observation_type = self._pred_type_from_box(observation_space)

            return observation_space, observation_type

        raise ValueError(f"Unimplemented: {observation_space.__class__.__name__}")

    def _pred_type_from_box(self, space: gym.spaces.Box):
        if not self.prediction_by_simulation:
            return EnvObservationType.UNKOWN

        # 実際の値を取得して予測
        is_discrete = True
        done = True
        for _ in range(100):
            if done:
                state = self.env.reset()
                if "int" not in str(np.asarray(state).dtype):
                    is_discrete = False
                    break
            va = self.env.fetch_valid_actions()
            if va is None:
                action = self.env.action_space.sample()
            else:
                action = random.choice(va)
            state, reward, done, _ = self.env.step(action)
            if "int" not in str(np.asarray(state).dtype):
                is_discrete = False
                break

        if is_discrete:
            return EnvObservationType.DISCRETE
        else:
            return EnvObservationType.CONTINUOUS

    def observation_encode(self, observation):
        observation = np.asarray(observation)
        if self.change_type == "Discrete->Box":
            return observation[np.newaxis, ...]
        if self.change_type == "Tuple->Box":
            return observation
        return observation


if __name__ == "__main__":
    pass
