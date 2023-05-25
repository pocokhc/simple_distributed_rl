from typing import Tuple

import numpy as np

from srl.base.define import EnvObservationTypes, RLObservationTypes
from srl.base.env.base import EnvRun
from srl.base.rl.processor import Processor
from srl.base.spaces import BoxSpace
from srl.base.spaces.space import SpaceBase


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        rl_observation_type: RLObservationTypes,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        observation_space = BoxSpace(
            low=-8,
            high=8,
            shape=(3, 1, 1),
        )
        return observation_space, EnvObservationTypes.SHAPE3

    def process_observation(self, observation: np.ndarray, env) -> np.ndarray:
        return observation.reshape((-1, 1, 1))
