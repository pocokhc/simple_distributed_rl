from dataclasses import dataclass, field
from typing import Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces.box import BoxSpace


@dataclass
class RenderImageProcessor(Processor):
    render_kwargs: dict = field(default_factory=lambda: {})

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: EnvRun,
        rl_config: RLConfig,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        env.reset()
        rgb_array = env.render_rgb_array(**self.render_kwargs)
        self.shape = rgb_array.shape

        new_space = BoxSpace(self.shape, 0, 255)
        return new_space, EnvObservationTypes.COLOR

    def preprocess_observation(
        self,
        observation: EnvObservationType,
        env: EnvRun,
    ) -> EnvObservationType:
        return env.render_rgb_array(**self.render_kwargs)
