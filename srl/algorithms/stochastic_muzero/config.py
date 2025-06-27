from dataclasses import dataclass, field
from typing import List, Optional

from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

"""
Paper
https://openreview.net/forum?id=X6D9bAHhBQ1
"""


@dataclass
class Config(RLConfig):
    num_simulations: int = 20
    discount: float = 0.999

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: (
            LRSchedulerConfig().set_step(100_000, 0.0001)  #
        )
    )

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: カテゴリ化する範囲
    reward_range: tuple = (-10, 10)
    reward_range_num: int = 100
    #: カテゴリ化する範囲
    value_range: tuple = (-10, 10)
    value_range_num: int = 100

    test_policy_tau: float = 0.1
    # policyの温度パラメータのリスト
    policy_tau: Optional[float] = None
    #: <:ref:`SchedulerConfig`>
    policy_tau_scheduler: SchedulerConfig = field(
        default_factory=lambda: (
            SchedulerConfig(default_scheduler=True)  #
            .add_constant(1.0, 50_000)
            .add_constant(0.5, 25_000)
            .add_constant(0.25)
        )
    )

    # td_steps: int = 5      # multisteps
    unroll_steps: int = 5  # unroll_steps
    codebook_size: int = 32  # codebook

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_fraction: float = 0.1
    root_dirichlet_adaptive: bool = False

    # PUCT
    c_base: float = 19652
    c_init: float = 1.25

    # model
    dynamics_blocks: int = 15
    commitment_cost: float = 0.25  # VQ_VAEのβ
    weight_decay: float = 0.0001
    weight_decay_afterstate: float = 0.001  # 強めに掛けたほうが安定する気がする

    # rescale
    enable_rescale: bool = False

    def get_name(self) -> str:
        return "StochasticMuZero"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_image_block.get_processors()

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.value_range[0] < self.value_range[1]):
            raise ValueError(f"assert {self.value_range[0]} < {self.value_range[1]}")
        if not (self.reward_range[0] < self.reward_range[1]):
            raise ValueError(f"assert {self.reward_range[0]} < {self.reward_range[1]}")
        if not (self.unroll_steps > 0):
            raise ValueError(f"assert {self.unroll_steps} > 0")
