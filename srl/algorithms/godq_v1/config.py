import logging
from dataclasses import dataclass, field
from typing import List, Literal

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig
from srl.rl.models.config.input_multi_block import InputMultiBlockConfig
from srl.rl.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class DenoiserConfig:
    condition_channels: int = 256
    channels_list: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    res_block_num_list: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_attention_list: list[bool] = field(default_factory=lambda: [False, False, False, False])
    use_bottleneck_attention: bool = True
    # edm
    noise_mean: float = -0.4
    noise_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20
    # sigma_offset_noise: float = 0.3
    sigma_data: float = 0.5
    # train
    lr: float = 1e-4
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_grad_norm: float = 1


@dataclass
class SamplerConfig:
    num_steps_denoising: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_min: float = 0
    s_max: float = float("inf")
    s_noise: float = 1


@dataclass
class Config(RLConfig[DiscreteSpace, MultiSpace[BoxSpace]]):
    # --- policy
    test_epsilon: float = 0
    test_policy: Literal["q", "int"] = "q"
    epsilon: float = 0.01

    # --- archive
    enable_archive: bool = False
    archive_steps: int = 200
    archive_max_size: int = 10
    archive_rate: float = 0.5
    search_max_step: int = 500
    archive_rankbase_alpha: float = 1.0

    # --- encoder/feat
    input_block: InputMultiBlockConfig = field(default_factory=lambda: InputMultiBlockConfig())
    feat_type: Literal["", "BYOL"] = "BYOL"

    # --- q
    train_q: bool = True
    replay_ratio: int = 1
    reset_net_interval: int = 5000
    discount: float = 0.999
    max_discount_steps: int = 500
    align_loss_coeff: float = 0.05
    enable_q_distribution: bool = True
    enable_q_rescale: bool = True

    # --- BYOL
    byol_model_update_rate: float = 0.1
    byol_model_update_interval: int = 10
    # --- int
    enable_int_q: bool = True
    int_rate: float = 0.5
    int_discount: float = 0.95
    int_align_loss_coeff: float = 0.1
    int_norm_momentum: float = 0.9
    int_reward_clip: float = 2.0
    int_reward_rnd_scale: float = 10.0
    int_reward_byol_scale: float = 1.0
    # --- int episodic
    enable_int_episodic: bool = False
    episodic_count_max: int = 100
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000

    # --- model/train
    base_units: int = 512
    max_grad_norm: float = 10
    batch_size: int = 64
    lr: float = 0.0001
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig(compress=False).set_replay_buffer())

    # --- diffusion
    enable_diffusion: bool = False
    train_diffusion: bool = True
    denoiser: DenoiserConfig = field(default_factory=lambda: DenoiserConfig())
    sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig())
    diff_lr: float = 0.0001

    def set_model(self, units: int):
        self.base_units = units
        self.input_block.cont_units = units
        self.input_block.discrete_units = units

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
        if self.feat_type not in ["", "BYOL"]:
            raise ValueError(f"assert {self.feat_type} in ['', 'BYOL']")

    def setup_from_env(self, env: EnvRun) -> None:
        if env.player_num != 1:
            raise ValueError(f"assert {env.player_num} == 1")

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.MULTI

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return self.input_block.get_processors(prev_observation_space, self)

    def use_render_image_state(self) -> bool:
        return self.enable_diffusion

    def get_render_image_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        return [ImageProcessor(image_type=SpaceTypes.RGB, resize=(64, 64), normalize_type="-1to1")]

    def use_update_parameter_from_worker(self) -> bool:
        return self.enable_archive
