from dataclasses import dataclass, field
from typing import List

from srl.base.rl.algorithms.base_ppo import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.hidden_block import HiddenBlockConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
Paper
https://arxiv.org/abs/1812.05905

DDPG
    Replay buffer       : o
    Target Network(soft): o
    Target Network(hard): o
    Add action noise    : x
TD3
    Clipped Double Q learning : o
    Target Policy Smoothing   : x
    Delayed Policy Update     : x
SAC
    Squashed Gaussian Policy: o
"""


@dataclass
class Config(RLConfig):
    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`HiddenBlockConfig`> policy layer
    policy_hidden_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64, 64, 64)))
    #: <:ref:`HiddenBlockConfig`>
    q_hidden_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((128, 128, 128)))

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: discount
    discount: float = 0.99
    #: policy learning rate
    lr_policy: float = 0.0001
    #: <:ref:`LRSchedulerConfig`>
    lr_policy_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: q learning rate
    lr_q: float = 0.0001
    #: <:ref:`LRSchedulerConfig`>
    lr_q_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: alpha learning rate
    lr_alpha: float = 0.0001
    #: <:ref:`LRSchedulerConfig`>
    lr_alpha_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: soft_target_update_tau
    soft_target_update_tau: float = 0.02
    #: hard_target_update_interval
    hard_target_update_interval: int = 10000
    #: actionが連続値の時、正規分布をtanhで-1～1に丸めるか
    enable_normal_squashed: bool = True

    start_steps: int = 10000
    #: entropy alphaを自動調整するか
    entropy_alpha_auto_scale: bool = True
    #: entropy alphaの初期値
    entropy_alpha: float = 0.2

    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_scale_range: tuple = (1e-10, 10)

    def get_name(self) -> str:
        return "SAC"

    def get_framework(self) -> str:
        return "tensorflow"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)
