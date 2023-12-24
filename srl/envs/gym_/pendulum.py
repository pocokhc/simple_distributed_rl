from typing import Tuple

import numpy as np

from srl.base.define import EnvObservationTypes, RLBaseTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.processor import Processor
from srl.base.spaces import BoxSpace
from srl.base.spaces.space import SpaceBase


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        rl_observation_type: RLBaseTypes,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        observation_space = BoxSpace(
            low=-8,
            high=8,
            shape=(1, 1, 3),
        )
        return observation_space, EnvObservationTypes.IMAGE

    def process_observation(self, observation: np.ndarray, env) -> np.ndarray:
        return observation.reshape((1, 1, -1))
