from dataclasses import dataclass
from typing import Generic, List, Union

import numpy as np

from srl.base.define import RLBaseTypes, TConfig, TParameter
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[Union[DiscreteSpace, ArrayContinuousSpace], BoxSpace]):
    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        Union[DiscreteSpace, ArrayContinuousSpace],
        Union[int, List[float]],
        BoxSpace,
        np.ndarray,
    ],
):
    pass
