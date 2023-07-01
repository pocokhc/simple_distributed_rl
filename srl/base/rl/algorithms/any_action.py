from dataclasses import dataclass

from srl.base.define import RLTypes
from srl.base.rl.base import RLConfig
from srl.base.rl.worker_rl import RLWorker


@dataclass
class AnyActionConfig(RLConfig):
    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.ANY


class AnyActionWorker(RLWorker):
    pass
