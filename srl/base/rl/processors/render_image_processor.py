import logging
from dataclasses import dataclass, field
from typing import Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes, RLTypes
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.processor import Processor
from srl.base.spaces.box import BoxSpace

logger = logging.getLogger(__name__)


@dataclass
class RenderImageProcessor(Processor):
    render_kwargs: dict = field(default_factory=lambda: {})

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        rl_observation_type: RLTypes,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        env.reset()
        rgb_array = env.render_rgb_array(**self.render_kwargs)
        self.shape = rgb_array.shape

        new_space = BoxSpace(self.shape, 0, 255)
        return new_space, EnvObservationTypes.COLOR

    def process_observation(
        self,
        observation: EnvObservationType,
        env: EnvRun,
    ) -> EnvObservationType:
        return env.render_rgb_array(**self.render_kwargs)
