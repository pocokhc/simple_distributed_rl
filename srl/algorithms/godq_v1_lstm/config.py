import logging
from dataclasses import dataclass, field
from typing import Literal

from srl.algorithms.godq_v1.config import Config as godq_config
from srl.rl.memories.episode_replay_buffer import EpisodeReplayBufferConfig

logger = logging.getLogger(__name__)


@dataclass
class Config(godq_config):
    encode_discrete_type: Literal["BOX", "Discrete", "Conv1D"] = "Conv1D"
    feat_type: Literal["", "SimSiam", "BYOL"] = "BYOL"

    batch_size: int = 64
    batch_length: int = 5
    lr: float = 0.0002
    memory: EpisodeReplayBufferConfig = field(default_factory=lambda: EpisodeReplayBufferConfig(compress=False))

    def get_name(self) -> str:
        return "GoDQ_v1_LSTM"
