from dataclasses import dataclass, field
from typing import List

from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.functions import create_epsilon_list
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_block import InputBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
・Paper
https://openreview.net/forum?id=r1lyTjAqYX

DQN
    window_length          : -
    Fixed Target Q-Network : o
    Error clipping     : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : x
    Reward clip        : x
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning         : x
    Noisy Network               : x
    Categorical DQN             : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : o
    Value function rescaling : o (config selection)
Never Give Up(NGU)
    Retrace          : o (config selection)
Other
    invalid_actions : o

"""


@dataclass
class Config(RLConfig):
    test_epsilon: float = 0
    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(default_factory=lambda: InputBlockConfig())
    lstm_units: int = 512
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig().set_dueling_network((512,)))

    # lstm
    burnin: int = 5
    sequence_length: int = 5

    discount: float = 0.997
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    target_model_update_interval: int = 1000

    # double dqn
    enable_double_dqn: bool = True

    # rescale
    enable_rescale: bool = False

    # retrace
    enable_retrace: bool = True
    retrace_h: float = 1.0

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(
            actor_num,
            epsilon=self.actor_epsilon,
            alpha=self.actor_alpha,
        )[actor_id]

    def set_atari_config(self):
        # model
        self.lstm_units = 512
        self.input_block.image.set_dqn_block()
        self.hidden_block.set_dueling_network((512,))

        # lstm
        self.burnin = 40
        self.sequence_length = 80

        self.discount = 0.997
        self.lr = 0.0001
        self.batch_size = 64
        self.target_model_update_interval = 2500

        self.enable_double_dqn = True
        self.enable_rescale = True
        self.enable_retrace = False

        self.memory.capacity = 1_000_000
        self.memory.set_proportional(
            alpha=0.9,
            beta_initial=0.6,
            beta_steps=1_000_000,
        )

    def get_name(self) -> str:
        return "R2D2"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space)

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.burnin >= 0):
            raise ValueError(f"assert {self.burnin} >= 0")
        if not (self.sequence_length >= 1):
            raise ValueError(f"assert {self.sequence_length} >= 1")
