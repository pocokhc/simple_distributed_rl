from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor

"""
ref: https://github.com/eloialonso/diamond/tree/main
"""


@dataclass
class DenoiserConfig:
    num_steps_conditioning: int = 4
    condition_channels: int = 256
    channels_list: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    res_block_num_list: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_attention_list: list[bool] = field(default_factory=lambda: [False, False, False, False])
    # edm
    noise_mean: float = -0.4
    noise_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20
    sigma_offset_noise: float = 0.3
    sigma_data: float = 0.5
    # train
    lr: float = 1e-4
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_grad_norm: float = 1


@dataclass
class DiffusionSamplerConfig:
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
class RewardEndModelConfig:
    lstm_dim: int = 512
    condition_channels: int = 128
    channels_list: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    res_block_num_list: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_attention_list: list[bool] = field(default_factory=lambda: [False, False, False, False])
    lr: float = 1e-4
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_grad_norm: float = 100


@dataclass
class ActorCriticConfig:
    lstm_dim: int = 512
    channels_list: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    enable_downsampling_list: list[bool] = field(default_factory=lambda: [True, True, True, True])
    lr: float = 1e-4
    weight_decay: float = 0
    eps: float = 1e-8
    max_grad_norm: float = 100


@dataclass
class Config(RLConfig):
    denoiser_cfg: DenoiserConfig = field(default_factory=lambda: DenoiserConfig())
    sampler_cfg: DiffusionSamplerConfig = field(default_factory=lambda: DiffusionSamplerConfig())
    actor_critic_cfg: ActorCriticConfig = field(default_factory=lambda: ActorCriticConfig())
    reward_end_cfg: RewardEndModelConfig = field(default_factory=lambda: RewardEndModelConfig())

    train_diffusion: bool = True
    train_reward_end: bool = True
    train_actor_critic: bool = True

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    burnin: int = 4
    horizon: int = 15

    discount: float = 0.985
    lambda_: float = 0.95

    weight_entropy_loss: float = 0.001
    weight_value_loss: float = 1.0

    img_shape: Tuple[int, int] = (64, 64)
    img_color: bool = True

    def set_small_params(self):
        self.img_shape = (64, 64)
        self.img_color = True
        self.denoiser_cfg = DenoiserConfig(
            num_steps_conditioning=1,
            condition_channels=128,
            channels_list=[64, 64, 64, 64],
            res_block_num_list=[2, 2, 2, 2],
            use_attention_list=[False, False, False, False],
            lr=0.0001,
        )
        self.reward_end_cfg = RewardEndModelConfig(
            lstm_dim=32,
            condition_channels=32,
            channels_list=[16, 16],
            res_block_num_list=[2, 2],
            use_attention_list=[False, False],
            lr=0.0001,
        )
        self.actor_critic_cfg = ActorCriticConfig(
            lstm_dim=32,
            channels_list=[16, 16, 32, 32],
            enable_downsampling_list=[True, True, True, True],
            lr=0.0001,
        )
        self.sampler_cfg = DiffusionSamplerConfig(
            num_steps_denoising=3,
        )
        self.batch_size = 32
        self.burnin = 1
        self.horizon = 3
        self.memory.capacity = 10000
        self.memory.warmup_size = 1000
        return self

    def get_name(self) -> str:
        return "DIAMOND"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        assert prev_observation_space.is_image(), "image only"
        self._processor = ImageProcessor(
            image_type=SpaceTypes.COLOR if self.img_color else SpaceTypes.GRAY_3ch,
            resize=self.img_shape,
            normalize_type="-1to1",
        )
        return [self._processor]

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.denoiser_cfg.num_steps_conditioning > 0):
            raise ValueError(f"assert {self.denoiser_cfg.num_steps_conditioning} > 0")
        if not (self.burnin >= 0):
            raise ValueError(f"assert {self.burnin} >= 0")
        if not (self.horizon >= 0):
            raise ValueError(f"assert {self.horizon} >= 0")

    def decode_img(self, img, scale: float = 2):
        import cv2

        img = np.clip(img, -1.0, 1.0)
        img = (((img + 1) / 2) * 255).astype(np.uint8)
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        return img
