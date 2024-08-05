from dataclasses import dataclass
from typing import Generic

import numpy as np

from srl.base.define import RLBaseActTypes, RLBaseObsTypes, TConfig, TParameter
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class RLConfig(RLConfigBase[DiscreteSpace, BoxSpace]):
    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.BOX


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        DiscreteSpace,
        int,
        BoxSpace,
        np.ndarray,
    ],
):
    pass
