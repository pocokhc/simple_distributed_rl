import logging
from dataclasses import dataclass, field
from typing import List, Literal

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    # --- archive
    enable_archive: bool = True
    archive_rate: float = 0.9
    search_max_step: int = 200
    archive_min_num: int = 5
    archive_max_size: int = 100
    archive_novelty_threshold: float = 0.1

    # --- latent
    latent_size: int = 8

    # --- int
    episodic_count_max: int = 10
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000
    enable_int_reward_debug: bool = False

    # --- encoder/feat
    encode_img_type: Literal["DQN", "R2D3"] = "DQN"
    feat_type: Literal["", "SimSiam", "SPR"] = "SPR"
    used_discrete_block: bool = True

    # --- SPR
    replay_ratio: int = 4
    reset_interval_head: int = 5001
    reset_interval_shrink: int = 50001
    select_model: Literal["online", "target"] = "target"

    # --- q policy
    test_epsilon: float = 0
    epsilon: float = 0.05

    # --- q train
    enable_reward_symlog_scalar: bool = True
    discount: float = -1  # -1 is auto. 0.999
    target_model_update_rate: float = 0.01
    init_target_q_zero: bool = True
    q_penalty_rate: float = 0  # 0.01
    reward_penalty: float = -0.01

    # --- model/train
    base_units: int = 512
    max_grad_norm: float = 5
    batch_size: int = 64
    lr: float = 0.0001
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig(compress=False).set_proportional_cpp(beta_steps=100_000))

    @property
    def enable_int_reward(self) -> bool:
        if self.enable_int_reward_debug:
            return True
        return self.enable_archive

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

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.BOX_UNTYPED

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            if self.encode_img_type == "DQN":
                return [ImageProcessor(SpaceTypes.GRAY_3ch, (84, 84), normalize_type="-1to1")]
            elif self.encode_img_type == "R2D3":
                return [ImageProcessor(SpaceTypes.COLOR, (96, 72), normalize_type="0to1")]
        return []

    def use_update_parameter_from_worker(self) -> bool:
        return True
