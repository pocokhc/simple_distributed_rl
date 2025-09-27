from dataclasses import dataclass, field
from typing import List, Literal

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
vae ref: https://developers-jp.googleblog.com/2019/04/tensorflow-probability-vae.html
ref: https://github.com/zacwellmer/WorldModels
"""


@dataclass
class Config(RLConfig):
    train_mode: Literal["vae", "rnn", "controller"] = "vae"

    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    batch_size: int = 32
    capacity: int = 100_000
    warmup_size: int = 100

    # VAE
    z_size: int = 32
    kl_tolerance: float = 0.5

    # MDN-RNN
    sequence_length: int = 10
    rnn_units: int = 256
    num_mixture: int = 5  # number of mixtures in MDN
    temperature: float = 1.15

    # GA
    num_simulations: int = 16
    num_individual: int = 16
    mutation: float = 0.01
    randn_sigma: float = 1.0
    blx_a: float = 0.1

    def get_changeable_parameters(self) -> List[str]:
        return [
            "train_mode",
            "memory_warmup_size",
            "kl_tolerance",
            "num_simulations",
            "num_individual",
            "mutation",
            "randn_sigma",
            "blx_a",
        ]

    def get_name(self) -> str:
        return "WorldModels"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image_like():
            if not (self.window_length == 1):
                raise ValueError(f"assert {self.window_length} == 1")
            return [
                ImageProcessor(
                    image_type=SpaceTypes.RGB,
                    resize=(64, 64),
                    normalize_type="0to1",
                )
            ]
        return []

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.temperature >= 0):
            raise ValueError(f"assert {self.temperature} >= 0")
