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
https://arxiv.org/abs/1911.08265

Ref
https://arxiv.org/src/1911.08265v2/anc/pseudocode.py

reanalyze : x
"""


@dataclass
class Config(RLConfig):
    #: シミュレーション回数
    num_simulations: int = 50
    #: 割引率
    discount: float = 0.99

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: (LRSchedulerConfig().set_step(100_000, 0.0001)))
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
    # td_steps: int = 5  # MC法でエピソード最後まで展開しているので未使用
    #: unroll_steps
    unroll_steps: int = 3

    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25

    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25

    #: Dynamics networkのブロック数
    dynamics_blocks: int = 15
    #: reward dense units
    reward_dense_units: int = 0
    #: weight decay
    weight_decay: float = 0.0001

    #: rescale
    enable_rescale: bool = False
    #: reanalyze
    enable_reanalyze: bool = False

    def set_atari_config(self):
        self.num_simulations = 50
        self.batch_size = 1024
        self.memory.warmup_size = 10_000
        self.discount = 0.997
        self.lr = 0.05
        self.lr_scheduler.set_step(350_000, 0.005)
        self.reward_range = (-300, 300)
        self.reward_range_num = 601
        self.value_range = (-300, 300)
        self.value_range_num = 601
        # self.td_steps = 10
        self.unroll_steps = 5
        self.policy_tau_scheduler.clear()
        self.policy_tau_scheduler.add_constant(1.0, 500_000)
        self.policy_tau_scheduler.add_constant(0.5, 250_000)
        self.policy_tau_scheduler.add_constant(0.25)
        self.input_image_block.set_muzero_atari_block(filters=128)
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True
        return self

    def set_board_game_config(self):
        self.num_simulations = 800
        self.batch_size = 2048
        self.memory.warmup_size = 10_000
        self.discount = 1.0
        self.lr = 0.05
        self.lr_scheduler.set_step(400_000, 0.005)
        self.reward_range = (-300, 300)
        self.reward_range_num = 601
        self.value_range = (-300, 300)
        self.value_range_num = 601
        # self.td_steps = 10
        self.unroll_steps = 10
        self.policy_tau_scheduler.clear()
        self.policy_tau_scheduler.add_constant(1.0, 500_000)
        self.policy_tau_scheduler.add_constant(0.5, 250_000)
        self.policy_tau_scheduler.add_constant(0.25)
        self.input_image_block.set_alphazero_block()
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True
        return self

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_image_block.get_processors()

    def get_name(self) -> str:
        return "MuZero"

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
