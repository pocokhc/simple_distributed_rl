import logging
from dataclasses import dataclass, field
from typing import List, Literal

from srl.base.define import SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    # --- policy
    test_epsilon: float = 0
    test_policy: Literal["q", "int"] = "q"
    epsilon: float = 0.05

    # --- archive
    enable_archive: bool = False
    archive_steps: int = 200
    archive_max_size: int = 10
    archive_rate: float = 0.5
    search_max_step: int = 500
    archive_rankbase_alpha: float = 1.0

    # --- encoder/feat
    encode_img_type: Literal["DQN", "R2D3"] = "DQN"
    encode_discrete_type: Literal["BOX", "Discrete", "Conv1D"] = "Discrete"
    enable_state_norm: bool = True
    used_discrete_block: bool = True
    feat_type: Literal["", "SimSiam", "BYOL"] = "SimSiam"

    # --- BYOL
    byol_model_update_rate: float = 0.1
    byol_model_update_interval: int = 10
    # --- int
    enable_int_q: bool = True
    int_target_prob: float = 0.9
    int_discount: float = 0.99
    int_align_loss_coeff: float = 0.03
    int_q_distribution: bool = False

    # --- q train
    replay_ratio: int = 2
    reset_net_interval: int = 5000
    max_discount_steps: int = 500
    enable_reward_symlog_scalar: bool = True
    discount: float = 0.999
    align_loss_coeff: float = 0.05
    enable_q_distribution: bool = True

    # --- model/train
    base_units: int = 256
    max_grad_norm: float = 5
    batch_size: int = 64
    lr: float = 0.0001
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig(compress=False).set_replay_buffer())

    def get_name(self) -> str:
        return "GoDQ_v1"

    def get_framework(self) -> str:
        return "torch"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.replay_ratio > 0):
            raise ValueError(f"assert {self.replay_ratio} > 0")
        if not (self.base_units >= 8):
            raise ValueError(f"assert {self.base_units} >= 8")

    def setup_from_env(self, env: EnvRun) -> None:
        if env.player_num != 1:
            raise ValueError(f"assert {env.player_num} == 1")

    def get_base_observation_type_options(self) -> SpaceEncodeOptions:
        return SpaceEncodeOptions(cast=False)

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            if self.encode_img_type == "DQN":
                return [ImageProcessor(SpaceTypes.GRAY_3ch, (84, 84), normalize_type="-1to1")]
            elif self.encode_img_type == "R2D3":
                return [ImageProcessor(SpaceTypes.COLOR, (96, 72), normalize_type="0to1")]
        return []

    def use_update_parameter_from_worker(self) -> bool:
        return self.enable_archive
