from dataclasses import dataclass
from typing import Generic, List

import numpy as np

from srl.base.define import RLBaseTypes, TConfig, TParameter
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.box import BoxSpace


@dataclass
class RLConfig(RLConfigBase[ArrayContinuousSpace, BoxSpace]):
    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        ArrayContinuousSpace,
        List[float],
        BoxSpace,
        np.ndarray,
    ],
):
    pass
