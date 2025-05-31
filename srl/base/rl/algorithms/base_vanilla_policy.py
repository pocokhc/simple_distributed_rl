from dataclasses import dataclass
from typing import Generic, List, Union

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import TRLMemory
from srl.base.rl.parameter import TRLParameter
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[Union[DiscreteSpace, ContinuousSpace], ArrayDiscreteSpace]):
    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.ARRAY_DISCRETE

    def get_framework(self) -> str:
        return ""


class RLWorker(
    Generic[TRLConfig, TRLParameter, TRLMemory],
    RLWorkerGeneric[
        TRLConfig,
        TRLParameter,
        TRLMemory,
        Union[DiscreteSpace, ContinuousSpace],
        Union[int, float],
        ArrayDiscreteSpace,
        List[int],
    ],
):
    pass
