import itertools
import logging
import random
from typing import List, Tuple

import numpy as np
from srl.base.define import (ContinuousAction, DiscreteAction,
                             DiscreteSpaceType, RLObservation)
from srl.base.env.base import SpaceBase

logger = logging.getLogger(__name__)


class ArrayDiscreteSpace(SpaceBase[List[int]]):
    def __init__(self, nvec: List[int]) -> None:
        self._nvec = nvec

        arr_list = [[n for n in range(v)] for v in nvec]

        if len(arr_list) > 10:
            logger.warn("It may take some time.")

        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i

    @property
    def nvec(self) -> List[int]:
        return self._nvec

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> List[int]:
        for _ in range(999):  # for safety
            r = [random.randint(0, n - 1) for n in self.nvec]
            if r not in invalid_actions:
                break
        return r

    def __eq__(self, o: object) -> bool:
        return self.nvec == o.nvec

    def __str__(self) -> str:
        return f"ArrayDiscrete({self.nvec})"

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return len(self.decode_tbl)

    def action_discrete_encode(self, val: List[int]) -> DiscreteAction:
        return self.encode_tbl[tuple(val)]

    def action_discrete_decode(self, val: DiscreteAction) -> List[int]:
        return list(self.decode_tbl[val])

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return len(self.nvec), np.array([0] * len(self.nvec)), np.array(self.nvec) - 1

    # def action_continuous_encode(self, val: List[int]) -> ContinuousAction:
    #    return [float(v) for v in val]

    def action_continuous_decode(self, val: ContinuousAction) -> List[int]:
        return [int(np.round(v)) for v in val]

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (len(self.nvec),)

    def observation_discrete_encode(self, val: List[int]) -> RLObservation:
        return np.array(val, dtype=int)

    def observation_continuous_encode(self, val: List[int]) -> RLObservation:
        return np.array(val, dtype=np.float32)
