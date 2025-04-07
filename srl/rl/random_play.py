import logging
from dataclasses import dataclass

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register_rulebase
from srl.base.rl.worker import RLWorker

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.NONE

    def get_name(self) -> str:
        return "random"


register_rulebase(Config(), __name__ + ":Worker")


class Worker(RLWorker):
    def policy(self, worker):
        return worker.sample_action()
