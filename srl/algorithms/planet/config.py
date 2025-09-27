from dataclasses import dataclass, field
from typing import Any, List, Literal

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
paper: https://arxiv.org/abs/1811.04551
ref: https://github.com/danijar/dreamer
"""


@dataclass
class Config(RLConfig):
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    batch_length: int = 50

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    # Model
    deter_size: int = 200
    stoch_size: int = 30
    num_units: int = 400
    dense_act: Any = "elu"
    cnn_act: Any = "relu"
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 1.0
    enable_overshooting_loss: bool = False
    max_overshooting_size: int = 5

    # GA
    action_algorithm: Literal["ga", "random"] = "ga"
    pred_action_length: int = 5
    num_generation: int = 10
    num_individual: int = 5
    num_simulations: int = 20
    mutation: float = 0.1
    print_ga_debug: bool = True

    # 経験取得方法
    experience_acquisition_method: Literal["episode", "loop"] = "episode"

    # other
    clip_rewards: Literal["none", "tanh"] = "none"
    dummy_state_val: float = 0.0

    def get_name(self) -> str:
        return "PlaNet"

    def get_framework(self) -> str:
        return "tensorflow"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        assert prev_observation_space.is_image_like(), "image only"
        return [
            ImageProcessor(
                image_type=SpaceTypes.RGB,
                resize=(64, 64),
                normalize_type="0to1",
            )
        ]
