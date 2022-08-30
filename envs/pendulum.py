from typing import Tuple

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.space import SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        observation_space = BoxSpace(
            low=-8,
            high=8,
            shape=(3, 1, 1),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, env) -> np.ndarray:
        return observation.reshape((-1, 1, 1))
