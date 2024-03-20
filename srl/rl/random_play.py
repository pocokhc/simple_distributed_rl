import logging
from typing import Tuple

from srl.base.define import RLActionType
from srl.base.rl.registration import register_rulebase
from srl.base.rl.worker import RLWorker

logger = logging.getLogger(__name__)


register_rulebase("random", __name__ + ":Worker")


class Worker(RLWorker):
    def policy(self, worker) -> Tuple[RLActionType, dict]:
        return worker.sample_action(), {}
