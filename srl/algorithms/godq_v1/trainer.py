import logging

from srl.base.rl.trainer import RLTrainer

from .config import Config
from .memory import Memory
from .parameter import Parameter
from .torch_trainer import TorchTrainer

logger = logging.getLogger(__name__)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        if self.config.enable_diffusion and self.config.train_diffusion:
            from .torch_diffusion_trainer import DiffusionTrainer

            self.trainer_diff = DiffusionTrainer()
            self.trainer_diff.on_setup(self)
        if self.config.train_q:
            self.trainer_q = TorchTrainer()
            self.trainer_q.on_setup(self)

    def train(self):
        if self.config.enable_diffusion and self.config.train_diffusion:
            self.trainer_diff.train()
            self.train_count = max(self.train_count, self.trainer_diff.train_count)
        if self.config.train_q:
            self.trainer_q.train()
            self.train_count = max(self.train_count, self.trainer_q.train_count)
