from dataclasses import dataclass
from typing import Generic, List, Union

from srl.base.define import RLBaseActTypes, RLBaseObsTypes, TConfig, TParameter
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[Union[DiscreteSpace, ArrayContinuousSpace], ArrayDiscreteSpace]):
    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE | RLBaseActTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.DISCRETE

    def get_framework(self) -> str:
        return ""


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        Union[DiscreteSpace, ArrayContinuousSpace],
        int,
        ArrayDiscreteSpace,
        List[int],
    ],
):
    pass
