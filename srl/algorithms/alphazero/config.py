from dataclasses import dataclass, field
from typing import List, Literal

from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.hidden_block import HiddenBlockConfig
from srl.rl.models.config.input_block import InputImageBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
Paper
AlphaGoZero: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
AlphaZero: https://arxiv.org/abs/1712.01815
           https://www.science.org/doi/10.1126/science.aar6404

Code ref:
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
"""


@dataclass
class Config(RLConfig):
    #: シミュレーション回数
    num_simulations: int = 100
    #: 割引率
    discount: float = 1.0
    #: エピソード序盤の確率移動のステップ数
    sampling_steps: int = 1

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: Learning rate
    lr: float = 0.002
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25

    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25

    #: <:ref:`InputImageBlockConfig`>
    input_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block(3, 64))
    #: <:ref:`HiddenBlockConfig`> value block
    value_block: HiddenBlockConfig = field(default_factory=lambda: HiddenBlockConfig().set((64,)))
    #: <:ref:`HiddenBlockConfig`> policy block
    policy_block: HiddenBlockConfig = field(default_factory=lambda: HiddenBlockConfig().set(()))

    #: "rate" or "linear"
    value_type: Literal["rate", "linear"] = "linear"

    def set_go_config(self):
        self.num_simulations = 800
        self.capacity = 500_000
        self.discount = 1.0
        self.sampling_steps = 30
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25
        self.batch_size = 4096
        self.memory.warmup_size = 10000
        self.lr_scheduler.set_piecewise(
            [300_000, 500_000],
            [0.02, 0.002, 0.0002],
        )
        self.input_block.set_alphazero_block(19, 256)
        self.value_block.set((256,))
        self.policy_block.set(())

    def set_chess_config(self):
        self.set_go_config()
        self.root_dirichlet_alpha = 0.3

    def set_shogi_config(self):
        self.set_go_config()
        self.root_dirichlet_alpha = 0.15

    def get_name(self) -> str:
        return "AlphaZero"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_block.get_processors()

    def get_framework(self) -> str:
        return "tensorflow"

    def use_backup_restore(self) -> bool:
        return True
