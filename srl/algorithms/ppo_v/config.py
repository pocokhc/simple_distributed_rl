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
    test_epsilon: float = 0.0
    policy_noise_normal_scale: float = 0.5

    #: Batch size
    batch_size: int = 64
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`HiddenBlockConfig`> hidden layers
    hidden_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64, 64)))
    #: <:ref:`HiddenBlockConfig`> value layers
    value_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64,)))
    #: <:ref:`HiddenBlockConfig`> policy layers
    policy_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set(()))

    #: discount
    discount: float = 0.95

    #: Clipped Surrogate Objective
    clip_range: float = 0.1

    #: Learning rate
    lr: float = 0.0002
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: エントロピーの反映率
    entropy_weight: float = 0

    loss_align_coeff: float = 0.1

    #: 勾配のL2におけるclip値
    max_grad_norm: float = 10

    #: actionが連続値の時、正規分布をtanhで-1～1に丸めるか
    squashed_gaussian_policy: bool = True
    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_scale_range: tuple = (1e-10, 10)

    def set_model(self, units: int = 64):
        self.hidden_block.set((units, units))
        self.value_block.set((units,))
        self.policy_block.set((units,))

    def get_name(self) -> str:
        return "V-PPO"

    def get_framework(self) -> str:
        return "torch"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)
