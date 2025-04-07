from dataclasses import dataclass
from typing import Generic, List

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import TRLMemory
from srl.base.rl.parameter import TRLParameter
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[DiscreteSpace, ArrayDiscreteSpace]):
    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.DISCRETE

    def get_framework(self) -> str:
        return ""


class RLWorker(
    Generic[TRLConfig, TRLParameter, TRLMemory],
    RLWorkerGeneric[
        TRLConfig,
        TRLParameter,
        TRLMemory,
        DiscreteSpace,
        int,
        ArrayDiscreteSpace,
        List[int],
    ],
):
    pass
