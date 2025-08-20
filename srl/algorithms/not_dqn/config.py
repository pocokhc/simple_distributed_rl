import logging
from dataclasses import dataclass, field
from typing import List

from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: ε-greedy parameter for Train
    epsilon: float = 0.1
    #: <:ref:`SchedulerConfig`>
    epsilon_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    #: Discount rate
    discount: float = 0.995

    #: 累積報酬和を計算する最大ステップ数
    max_n_step: int = 500
    #: Q値がNステップ割引累積報酬和から乖離しすぎないようにする正則化項の係数
    alignment_loss_coeff: float = 0.1
    #: <:ref:`SchedulerConfig`>
    alignment_loss_coeff_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())

    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: Batch size
    batch_size: int = 32
    #: Learning rate
    lr: float = 0.0002
    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    #: <:ref:`DuelingNetworkConfig`> hidden+out layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    def get_name(self) -> str:
        return "NoTarget_DQN"

    def get_framework(self) -> str:
        return "torch"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)

    def setup_from_env(self, env: EnvRun) -> None:
        if env.player_num != 1:
            raise ValueError(f"assert {env.player_num} == 1")


class Memory(RLPriorityReplayBuffer):
    pass
