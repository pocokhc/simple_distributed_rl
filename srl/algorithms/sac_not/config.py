from dataclasses import dataclass, field
from typing import List

from srl.base.rl.algorithms.base_ppo import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.hidden_block import HiddenBlockConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig


@dataclass
class Config(RLConfig):
    epsilon: float = 0.1
    test_epsilon: float = 0
    policy_training_scale: float = 0.5

    #: Batch size
    batch_size: int = 64
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`HiddenBlockConfig`> policy layers
    policy_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig())
    #: <:ref:`HiddenBlockConfig`> q layers
    q_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig())
    act_emb_units: int = 64

    #: Learning rate
    lr: float = 0.0001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: discount
    discount: float = 0.95

    target_policy_temperature: float = 2.0
    #: Target policy ノイズの標準偏差
    target_policy_noise_stddev: float = 0.2
    #: Target policy ノイズのclip範囲
    target_policy_clip_range: float = 0.5

    loss_align_coeff: float = 0.1
    max_grad_norm: float = 10.0

    squashed_gaussian_policy: bool = True
    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_scale_range: tuple = (1e-10, 10)

    def set_model(self, units: int = 64):
        self.policy_block.set((units, units))
        self.q_block.set((units, units))
        self.act_emb_units = units

    def get_name(self) -> str:
        return "NoT_SAC"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)

    def get_framework(self) -> str:
        return "torch"
