from dataclasses import dataclass
from typing import Tuple

from srl.base.define import RLActionType
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.memories.sequence_memory import SequenceMemory


@dataclass
class Config(DummyRLConfig):
    pass


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(SequenceMemory):
    pass


class Parameter(RLParameter):
    def call_restore(self, data, **kwargs) -> None:
        pass

    def call_backup(self, **kwargs):
        return None


class Trainer(RLTrainer):
    def train_on_batchs(self, memory_sample_return) -> None:
        self.train_count += 1


class Worker(RLWorker):
    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, dict]:
        return worker.sample_action(), {}
