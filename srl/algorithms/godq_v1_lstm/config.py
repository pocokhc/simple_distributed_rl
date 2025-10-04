import logging
from dataclasses import dataclass, field

from srl.algorithms.godq_v1.config import Config as godq_config
from srl.rl.memories.replay_buffer import ReplayBufferConfig

logger = logging.getLogger(__name__)


@dataclass
class Config(godq_config):
    lstm_c_clip: float = 10.0

    # --- int
    int_discount: float = 0.9
    int_align_loss_coeff: float = 0.1
    int_reward_clip: float = 2.0

    # --- train
    batch_size: int = 64
    batch_length: int = 1
    lr: float = 0.0001
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig(compress=False))

    def get_name(self) -> str:
        return "GoDQ_v1_LSTM"
