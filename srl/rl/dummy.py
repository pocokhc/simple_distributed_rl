import logging
from typing import Tuple

from srl.base.define import EnvAction, RLActionType, RLObservationType
from srl.base.rl.algorithms.modelbase import ModelBaseWorker
from srl.base.rl.base import RLConfig, RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.sequence_memory import SequenceRemoteMemory

logger = logging.getLogger(__name__)


class Config(RLConfig):
    def __init__(self) -> None:
        super().__init__()

    @property
    def action_type(self) -> RLActionType:
        return RLActionType.ANY

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.ANY

    @staticmethod
    def getName() -> str:
        return "Dummy"


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class RemoteMemory(SequenceRemoteMemory):
    pass


class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)

    def call_restore(self, data, **kwargs) -> None:
        pass

    def call_backup(self, **kwargs):
        return None


class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        self.train_count += 1
        return {}


class Worker(ModelBaseWorker):
    def __init__(self, *args):
        super().__init__(*args)

    def call_on_reset(self, state, env, worker) -> dict:
        return {}

    def call_policy(self, state, env, worker) -> Tuple[EnvAction, dict]:
        return env.sample(), {}

    def call_on_step(self, next_state, reward: float, done: bool, env, worker):
        return {}
