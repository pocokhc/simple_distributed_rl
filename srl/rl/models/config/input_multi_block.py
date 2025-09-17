from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, cast

import numpy as np

from srl.base.define import EnvObservationType, RLBaseTypes, SpaceTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions
from srl.rl.functions import image_processor


@dataclass
class InputMultiBlockConfig:
    image_type: Literal["DQN", "R2D3"] = "DQN"
    image_normalize_type: Literal["0to1", "-1to1"] = "-1to1"

    discrete_type: Literal["Discrete", "BOX", "Conv1D"] = "Discrete"
    discrete_target_params: int = 4096 * 4
    discrete_low_units: int = 4
    discrete_units: int = 64

    cont_units: int = 64

    def get_processors(self, prev_observation_space: SpaceBase, rl_config: RLConfig) -> List[RLProcessor]:
        return [InputMultiBlockProcessor(self, rl_config.get_dtype("np"))]

    def create_tf_block(self, cfg: RLConfig, out_flatten: bool = True, rnn: bool = False, **kwargs):
        raise ValueError("TODO")

    def create_tf_dummy_data(self, cfg: RLConfig, batch_size: int = 1, timesteps: int = 0) -> np.ndarray:
        raise ValueError("TODO")

    def create_torch_block(self, in_space: MultiSpace):
        from srl.rl.torch_.blocks.input_multi_block import InputMultiBlock

        return InputMultiBlock(self, in_space)


@dataclass
class InputMultiBlockProcessor(RLProcessor):
    cfg: InputMultiBlockConfig = field(default_factory=lambda: InputMultiBlockConfig())
    dtype: Any = np.float32

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        self.multi_space = cast(
            MultiSpace,
            prev_space.set_encode_space(RLBaseTypes.MULTI, SpaceEncodeOptions()),
        )

        new_spaces = []
        for space in self.multi_space.spaces:
            if space.is_image():
                if self.cfg.image_type == "DQN":
                    stype = SpaceTypes.GRAY_3ch
                    shape = (84, 84, 1)
                elif self.cfg.image_type == "R2D3":
                    stype = SpaceTypes.COLOR
                    shape = (72, 96, 1)
                if self.cfg.image_normalize_type == "0to1":
                    low = 0
                    high = 1
                elif self.cfg.image_normalize_type == "-1to1":
                    low = -1
                    high = 1
                new_spaces.append(
                    BoxSpace(
                        shape,
                        low,
                        high,
                        self.dtype,
                        stype,
                    )
                )
            elif space.is_discrete() and (self.cfg.discrete_type in ["Discrete", "Conv1D"]):
                new_spaces.append(
                    space.set_encode_space(
                        RLBaseTypes.NP_ARRAY,
                        SpaceEncodeOptions(np_zero_start=True),
                    )
                )
            else:
                new_spaces.append(
                    space.set_encode_space(
                        RLBaseTypes.NP_ARRAY,
                        SpaceEncodeOptions(cast=True, cast_dtype=self.dtype),
                    )
                )
        return MultiSpace(new_spaces)

    def remap_observation(self, states: Any, prev_spaces: SpaceBase, new_spaces: SpaceBase, **kwargs) -> EnvObservationType:
        assert isinstance(new_spaces, MultiSpace)

        states = prev_spaces.encode_to_space(states)

        new_states = []
        for state, prev_space, new_space in zip(states, self.multi_space.spaces, new_spaces.spaces):
            if prev_space.is_image():
                if self.cfg.image_type == "DQN":
                    resize = (84, 84)
                elif self.cfg.image_type == "R2D3":
                    resize = (96, 72)
                new_states.append(
                    image_processor(
                        state,  # type: ignore
                        prev_space.stype,
                        new_space.stype,
                        resize,
                        normalize_type=self.cfg.image_normalize_type,
                        shape_order="HWC",
                    )
                )
            else:
                new_states.append(prev_space.encode_to_space(state))
        return new_states
