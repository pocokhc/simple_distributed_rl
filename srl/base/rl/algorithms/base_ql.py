from dataclasses import dataclass
from typing import Generic, List

from srl.base.define import RLBaseTypes, TConfig, TParameter
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[DiscreteSpace, ArrayDiscreteSpace]):
    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_framework(self) -> str:
        return ""


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        DiscreteSpace,
        int,
        ArrayDiscreteSpace,
        List[int],
    ],
):
    pass
