from dataclasses import dataclass, field
from typing import List, Optional

from srl.base.rl.algorithms.base_ppo import RLConfig, RLWorker  # noqa: F401 # use worker
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

"""
EfficientZero
https://arxiv.org/abs/2111.00210
https://github.com/YeWR/EfficientZero

EfficientZeroV2
https://arxiv.org/abs/2403.00564
https://github.com/Shengjiewang-Jason/EfficientZeroV2

gumbel search        : o
Gaussian distribution: o
reanalyze            : x
"""


@dataclass
class Config(RLConfig):
    # --- MCTS
    #: シミュレーション回数
    num_simulations: int = 50
    num_top_actions: int = 4
    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25
    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25
    soft_minmax_q_e: float = 0.01
    #: Gumbel search
    enable_gumbel_search: bool = True
    c_visit: int = 50
    c_scale: float = 0.1

    #: 割引率
    discount: float = 0.997

    #: Batch size
    batch_size: int = 256
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())
    use_max_priority: bool = True

    downsample: bool = False
    res_blocks: int = 1
    res_channels: int = 64
    normalize_momentam: float = 0.9  # torchとtfで逆 torch:0.1 == tf:0.9

    reward_units: int = 512
    reward_range: tuple = (-300, 300)
    reward_range_num: int = 601

    value_units: int = 512
    value_range: tuple = (-300, 300)
    value_range_num: int = 601
    policy_units: int = 512

    projection_hid: int = 256
    projection_out: int = 256
    projection_head_hid: int = 64
    projection_head_out: int = 256

    #: Learning rate
    lr: float = 0.2
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: (
            LRSchedulerConfig().set_step(120_000, 0.02)  #
        )
    )
    max_grad_norm: float = 5

    consistency_loss_coeff: float = 1
    policy_loss_coeff: float = 1
    value_loss_coeff: float = 1
    reward_loss_coeff: float = 1

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
    # td_steps: int = 5  # MC法でエピソード最後まで展開しているので未使用
    #: unroll_steps
    unroll_steps: int = 5

    #: rescale
    enable_rescale: bool = False
    #: reanalyze
    enable_reanalyze: bool = False

    def set_small_params(self):
        self.batch_size = 32
        self.memory.warmup_size = 1000
        self.memory.capacity = 100_000
        self.discount = 0.997

        self.res_blocks = 1
        self.res_channels = 64
        self.reward_units = 64
        self.reward_range = (-20, 20)
        self.reward_range_num = 100
        self.value_units = 64
        self.value_range = (-20, 20)
        self.value_range_num = 100
        self.policy_units = 64
        self.projection_hid = 64
        self.projection_out = 64
        self.projection_head_hid = 32
        self.projection_head_out = 64
        return self

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return []

    def get_name(self) -> str:
        return "EfficientZero"

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
        if self.enable_reanalyze:
            raise ValueError("Not implemented")
        if not (self.num_top_actions % 2 == 0):
            raise ValueError(f"assert {self.num_top_actions} % 2 == 0")
