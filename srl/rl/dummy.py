from dataclasses import dataclass
from typing import Tuple

from srl.base.define import RLActionType
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import DummyConfig
from srl.base.rl.registration import register
from srl.base.rl.worker_rl import RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.rl.memories.sequence_memory import SequenceRemoteMemory


@dataclass
class Config(DummyConfig):
    pass


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


class Worker(RLWorker):
    def call_policy(self, worker: WorkerRun) -> Tuple[RLActionType, dict]:
        return worker.sample_action(), {}
