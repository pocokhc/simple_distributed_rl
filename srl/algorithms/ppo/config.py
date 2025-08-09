from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from srl.base.rl.algorithms.base_ppo import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.models.config.hidden_block import HiddenBlockConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
Paper
https://arxiv.org/abs/1707.06347
https://arxiv.org/abs/2005.12729

Clipped Surrogate Objective : o
Adaptive KL Penalty         : o
GAE                         : o
Other
  Value Clipping : o
  Reward scaling : o
  Orthogonal initialization and layer scaling: x
  Adam learning rate annealing : o
  Reward Clipping              : o
  Observation Normalization    : o
  Observation Clipping         : o
  Hyperbolic tan activations   : x
  Global Gradient Clipping     : o
"""


@dataclass
class Config(RLConfig):
    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig(capacity=2000))

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`HiddenBlockConfig`> hidden layers
    hidden_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64, 64)))
    #: <:ref:`HiddenBlockConfig`> value layers
    value_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64,)))
    #: <:ref:`HiddenBlockConfig`> policy layers
    policy_block: HiddenBlockConfig = field(init=False, default_factory=lambda: HiddenBlockConfig().set((64,)))

    #: 割引報酬の計算方法
    #:
    #: Parameters:
    #:   "MC" : モンテカルロ法
    #:   "GAE": Generalized Advantage Estimator
    experience_collection_method: str = "GAE"
    #: discount
    discount: float = 0.9
    #: GAEの割引率
    gae_discount: float = 0.9

    #: baseline
    #:
    #: Parameters:
    #:  "" "none"       : none
    #:  "ave"           : (adv - mean)
    #:  "std"           : adv/std
    #:  "normal"        : (adv - mean)/std
    #:  "advantage" "v" : adv - v
    baseline_type: str = "advantage"
    #: surrogate type
    #:
    #: Parameters:
    #:  ""     : none
    #:  "clip" : Clipped Surrogate Objective
    #:  "kl"   : Adaptive KLペナルティ
    surrogate_type: str = "clip"
    #: Clipped Surrogate Objective
    policy_clip_range: float = 0.2
    #: Adaptive KLペナルティ内の定数
    adaptive_kl_target: float = 0.01

    #: value clip flag
    enable_value_clip: float = True
    #: value clip range
    value_clip_range: float = 0.2

    #: Learning rate
    lr: float = 0.02
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig().set_step(2000, 0.01))
    #: 状態価値の反映率
    value_loss_weight: float = 1.0
    #: エントロピーの反映率
    entropy_weight: float = 0.1

    #: 状態の正規化 flag
    enable_state_normalized: bool = False
    #: 勾配のL2におけるclip値(0で無効)
    global_gradient_clip_norm: float = 0.5
    #: 状態のclip(Noneで無効、(-10,10)で指定)
    state_clip: Optional[Tuple[float, float]] = None
    #: 報酬のclip(Noneで無効、(-10,10)で指定)
    reward_clip: Optional[Tuple[float, float]] = None

    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_scale_range: tuple = (1e-10, 10)

    def get_name(self) -> str:
        return "PPO"

    def get_framework(self) -> str:
        return "tensorflow"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)
