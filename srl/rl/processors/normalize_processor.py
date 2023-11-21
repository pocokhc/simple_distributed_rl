from dataclasses import dataclass
from typing import Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class NormalizeProcessor(Processor):
    feature_rang: Tuple[float, float] = (0, 1)

    def __post_init__(self):
        pass

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: EnvRun,
        rl_config: RLConfig,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        self.obs_min = env_observation_space.low
        self.obs_max = env_observation_space.high
        self.old_range = self.obs_max - self.obs_min
        self.new_range = self.feature_rang[1] - self.feature_rang[0]

        if isinstance(env_observation_space, DiscreteSpace):
            return ContinuousSpace(self.feature_rang[0], self.feature_rang[1]), EnvObservationTypes.CONTINUOUS

        if isinstance(env_observation_space, ContinuousSpace):
            return ContinuousSpace(self.feature_rang[0], self.feature_rang[1]), env_observation_type

        if isinstance(env_observation_space, ArrayDiscreteSpace):
            return (
                ArrayContinuousSpace(env_observation_space._size, self.feature_rang[0], self.feature_rang[1]),
                EnvObservationTypes.CONTINUOUS,
            )

        if isinstance(env_observation_space, ArrayContinuousSpace):
            return (
                ArrayContinuousSpace(env_observation_space._size, self.feature_rang[0], self.feature_rang[1]),
                env_observation_type,
            )

        if isinstance(env_observation_space, BoxSpace):
            return (
                BoxSpace(env_observation_space.shape, self.feature_rang[0], self.feature_rang[1]),
                env_observation_type,
            )

        return env_observation_space, env_observation_type

    def preprocess_observation(self, observation: EnvObservationType, env: EnvRun) -> EnvObservationType:
        return self.feature_rang[0] + ((observation - self.obs_min) / self.old_range) * self.new_range
