from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig


class DownSamplingProcessor(RLProcessor):
    def remap_observation_space(self, prev_space: SpaceBase, rl_config: "Config", **kwargs) -> Optional[SpaceBase]:
        return BoxSpace(rl_config.downsampling_size, 0, 8, np.uint8, SpaceTypes.GRAY_2ch)

    def remap_observation(
        self,
        state,
        prev_space: SpaceBase,
        new_space: SpaceBase,
        rl_config: "Config",
        **kwargs,
    ):
        # (1) color -> gray
        if prev_space.stype == SpaceTypes.COLOR:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        elif prev_space.stype == SpaceTypes.GRAY_3ch:
            state = np.squeeze(state, axis=-1)

        # (2) down sampling
        state = cv2.resize(
            state,
            (rl_config.downsampling_size[1], rl_config.downsampling_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # (3) 255->8
        state = np.round(rl_config.downsampling_val * (state / 255.0))
        return state.astype(np.uint8)


@dataclass
class Config(RLConfig):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    epsilon: float = 0.01

    # --- archive parameters
    action_change_rate: float = 0.05
    explore_max_step: int = 100
    demo_batch_rate: float = 0.1
    w_visit: float = 0.3
    w_select: float = 0
    w_total_select: float = 0.1
    eps1: float = 0.001
    eps2: float = 0.00001

    downsampling_size: tuple = (11, 8)
    downsampling_val: int = 8

    # --- q parameters
    memory_warmup_size: int = 1_000
    memory_capacity: int = 10_000
    lr: float = 0.0005
    batch_size: int = 32
    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 2000
    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    #: <:ref:`DuelingNetworkConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    def get_name(self) -> str:
        return "Go-Explore"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []

    def get_framework(self) -> str:
        return "tensorflow"

    def get_render_image_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return [DownSamplingProcessor()]

    def use_render_image_state(self) -> bool:
        return True

    def use_backup_restore(self) -> bool:
        return True

    def use_update_parameter_from_worker(self) -> bool:
        return True
