import logging
from dataclasses import dataclass
from typing import Tuple

from srl.base.define import EnvObservation, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class RenderImageProcessor(Processor):

    render_kwargs: dict = None

    def __post_init__(self):
        if self.render_kwargs is None:
            self.render_kwargs = {}

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:

        env.reset()
        rgb_array = env.render_rgb_array(**self.render_kwargs)
        self.shape = rgb_array.shape

        new_space = BoxSpace(self.shape, 0, 255)
        return new_space, EnvObservationType.COLOR

    def process_observation(
        self,
        observation: EnvObservation,
        env: EnvRun,
    ) -> EnvObservation:
        return env.render_rgb_array(**self.render_kwargs)
