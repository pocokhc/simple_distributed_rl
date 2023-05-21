import logging
from dataclasses import dataclass
from typing import Tuple

from srl.base.define import EnvActionType, RLActionTypes, RLObservationTypes
from srl.base.rl.algorithms.modelbase import ModelBaseWorker
from srl.base.rl.base import RLConfig, RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.sequence_memory import SequenceRemoteMemory

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    @property
    def action_type(self) -> RLActionTypes:
        return RLActionTypes.ANY

    @property
    def observation_type(self) -> RLObservationTypes:
        return RLObservationTypes.ANY

    def getName(self) -> str:
        return "Dummy"


register(
    Config(),
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

    def call_policy(self, state, env, worker) -> Tuple[EnvActionType, dict]:
        return env.sample(), {}

    def call_on_step(self, next_state, reward: float, done: bool, env, worker):
        return {}
