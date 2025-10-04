import logging

from srl.base.rl.trainer import RLTrainer

from .config import Config
from .memory import Memory
from .parameter import Parameter
from .torch_trainer import TorchTrainer

logger = logging.getLogger(__name__)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        self.trainer_q = TorchTrainer()
        self.trainer_q.on_setup(self)

    def train(self):
        self.trainer_q.train()
        self.train_count = max(self.train_count, self.trainer_q.train_count)
