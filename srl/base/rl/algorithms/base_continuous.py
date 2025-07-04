from dataclasses import dataclass
from typing import Generic

import numpy as np

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig as RLConfigBase
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import TRLMemory
from srl.base.rl.parameter import TRLParameter
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.np_array import NpArraySpace


@dataclass
class RLConfig(RLConfigBase[NpArraySpace, BoxSpace]):
    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.NP_ARRAY

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.BOX


class RLWorker(
    Generic[TRLConfig, TRLParameter, TRLMemory],
    RLWorkerGeneric[
        TRLConfig,
        TRLParameter,
        TRLMemory,
        NpArraySpace,
        np.ndarray,
        BoxSpace,
        np.ndarray,
    ],
):
    pass
