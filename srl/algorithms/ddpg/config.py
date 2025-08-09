from dataclasses import dataclass, field
from typing import List

from srl.base.rl.algorithms.base_continuous import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.hidden_block import HiddenBlockConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
Ref
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

DDPG
    Replay buffer       : o
    Target Network(soft): o
    Target Network(hard): o
    Add action noise    : o
TD3
    Clipped Double Q learning : o
    Target Policy Smoothing   : o
    Delayed Policy Update     : o
"""


@dataclass
class Config(RLConfig):
    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`HiddenBlockConfig`> policy layers
    policy_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig())
    #: <:ref:`HiddenBlockConfig`> q layers
    q_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig())

    #: Learning rate
    lr: float = 0.005
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: discount
    discount: float = 0.9
    #: soft_target_update_tau
    soft_target_update_tau: float = 0.02
    #: hard_target_update_interval
    hard_target_update_interval: int = 100

    #: ノイズ用の標準偏差
    noise_stddev: float = 0.2
    #: Target policy ノイズの標準偏差
    target_policy_noise_stddev: float = 0.2
    #: Target policy ノイズのclip範囲
    target_policy_clip_range: float = 0.5
    #: Actorの学習間隔
    actor_update_interval: int = 2

    def get_name(self) -> str:
        return "DDPG"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)

    def get_framework(self) -> str:
        return "tensorflow"
