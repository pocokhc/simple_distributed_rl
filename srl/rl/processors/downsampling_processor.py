from dataclasses import dataclass
from typing import Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType, SpaceTypes
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.utils.common import is_package_installed


@dataclass
class DownSamplingProcessor(RLProcessor):
    resize: Tuple[int, int] = (11, 8)  # (w, h)
    max_level: int = 8

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        if not is_package_installed("cv2"):
            return None
        if prev_space.stype not in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
        ]:
            return None
        return BoxSpace((self.resize[1], self.resize[0]), 0, self.max_level, np.uint8, SpaceTypes.GRAY_2ch)

    def remap_observation(self, state: EnvObservationType, prev_space: SpaceBase, new_space: SpaceBase, **kwargs) -> EnvObservationType:
        import cv2

        state = cast(np.ndarray, state)
        if prev_space.stype == SpaceTypes.GRAY_3ch:
            state = np.squeeze(state, axis=-1)
        elif prev_space.stype == SpaceTypes.COLOR:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        # (h,w) = cv2.resize((h,w), resize=(w, h))
        state = cv2.resize(state, self.resize, interpolation=cv2.INTER_NEAREST)

        state = np.round(self.max_level * (state / 255.0))
        return state.astype(np.uint8)

    def decode(self, img: np.ndarray, resize=(64, 64)):
        import cv2

        img = (np.round(img * self.max_level / 255.0) * 255 / self.max_level).astype(np.uint8)
        img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
        return img
