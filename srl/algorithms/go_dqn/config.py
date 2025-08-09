from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

# 初期値がランダムなDQN


class DownSamplingProcessor(RLProcessor):
    def remap_observation_space(self, prev_space: SpaceBase, rl_config: "Config", **kwargs) -> Optional[SpaceBase]:
        return BoxSpace(rl_config.downsampling_size, 0, 8, np.uint8, SpaceTypes.GRAY_2ch)

    def remap_observation(
        self,
        img,
        prev_space: SpaceBase,
        new_space: SpaceBase,
        rl_config: "Config",
        _debug=False,
        **kwargs,
    ):
        if prev_space.stype == SpaceTypes.COLOR:
            img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif prev_space.stype == SpaceTypes.GRAY_3ch:
            img1 = np.squeeze(img, axis=-1)
        else:
            img1 = img

        ret, img2 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("a", img2)
        # cv2.waitKey(0)

        img3 = cv2.resize(
            img2,
            (rl_config.downsampling_size[1], rl_config.downsampling_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        img3 = np.where(img3 < 127, 0, 1).astype(np.uint8)

        if _debug:
            return img1, img2, img3
        return img3


@dataclass
class Config(RLConfig):
    epsilon: float = 0.001
    test_epsilon: float = 0.00001

    restore_rate: float = 0.9

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(
        default_factory=lambda: ReplayBufferConfig(
            warmup_size=2_000,
            capacity=50_000,
        )
    )

    #: Learning rate
    lr: float = 0.0001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    downsampling_size: tuple = (12, 12)

    go_rate: float = 0.9
    go_action_change_rate: float = 0.05
    ucb_scale: float = 0.1
    search_max_step: int = 100

    #: Discount rate
    discount: float = 0.995
    #: Synchronization interval to Target network
    target_model_update_interval: int = 2000

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`MLPBlockConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    def get_name(self) -> str:
        return "GoDQN"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)

    def get_framework(self) -> str:
        return "tensorflow"

    def get_render_image_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return [DownSamplingProcessor()]

    def use_render_image_state(self) -> bool:
        return True

    def use_backup_restore(self) -> bool:
        return True
