import itertools
import logging
import random
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLActionType, RLObservation

from .space import SpaceBase

logger = logging.getLogger(__name__)


class ArrayDiscreteSpace(SpaceBase[List[int]]):
    def __init__(
        self,
        size: int,
        low: Union[int, List[int]],
        high: Union[int, List[int]],
    ) -> None:
        self._size = size
        assert isinstance(size, int)

        self._low = [low for _ in range(self.size)] if isinstance(low, int) else low
        assert len(self._low) == size
        self._low = [int(low) for low in self._low]

        self._high = [high for _ in range(self.size)] if isinstance(high, int) else high
        assert len(self._high) == size
        self._high = [int(h) for h in self._high]

        self.decode_tbl = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def low(self) -> List[int]:
        return self._low

    @property
    def high(self) -> List[int]:
        return self._high

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> List[int]:
        self._create_action_tbl()

        valid_actions = []
        for a in self.decode_tbl:  # decode_tbl is all action
            if a not in invalid_actions:
                valid_actions.append(a)

        return list(random.choice(valid_actions))

    def convert(self, val: Any) -> List[int]:
        if isinstance(val, list):
            return [int(np.round(v)) for v in val]
        elif isinstance(val, tuple):
            return [int(np.round(v)) for v in val]
        elif isinstance(val, np.ndarray):
            return val.round().astype(int).tolist()
        return [int(np.round(val)) for _ in range(self.size)]

    def __str__(self) -> str:
        return f"ArrayDiscrete({self.size}, range[{int(np.min(self.low))}, {int(np.max(self.high))}])"

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != self.size:
            return False
        for i in range(self.size):
            if not isinstance(val[i], int):
                return False
            if val[i] < self.low[i]:
                return False
            if val[i] > self.high[i]:
                return False
        return True

    @property
    def base_action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    def __eq__(self, o: "ArrayDiscreteSpace") -> bool:
        if self.size != o.size:
            return False
        if self.low is None:
            if o.low is not None:
                return False
        else:
            if o.low is None:
                return False
            for i in range(self.size):
                if self.low[i] != o.low[i]:
                    return False
        if self.high is None:
            if o.high is not None:
                return False
        else:
            if o.high is None:
                return False
            for i in range(self.size):
                if self.high[i] != o.high[i]:
                    return False
        return True

    # --- test
    def assert_params(self, true_size: int, true_low: List[int], true_high: List[int]):
        assert self.size == true_size
        assert self.low == true_low
        assert self.high == true_high

    # --- discrete
    def _create_action_tbl(self) -> None:
        if self.decode_tbl is not None:
            return

        if self.size > 10:
            logger.warning("It may take some time.")

        arr_list = [[a for a in range(self.low[i], self.high[i] + 1)] for i in range(self.size)]

        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        self._create_action_tbl()
        return len(self.decode_tbl)

    def action_discrete_encode(self, val: List[int]) -> DiscreteAction:
        self._create_action_tbl()
        return self.encode_tbl[tuple(val)]

    def action_discrete_decode(self, val: DiscreteAction) -> List[int]:
        self._create_action_tbl()
        return list(self.decode_tbl[val])

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return self.size, np.array(self.low), np.array(self.high)

    # def action_continuous_encode(self, val: List[int]) -> ContinuousAction:
    #    return [float(v) for v in val]

    def action_continuous_decode(self, val: ContinuousAction) -> List[int]:
        return [int(np.round(v)) for v in val]

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.size,)

    def observation_discrete_encode(self, val: List[int]) -> RLObservation:
        return np.array(val, dtype=int)

    def observation_continuous_encode(self, val: List[int]) -> RLObservation:
        return np.array(val, dtype=np.float32)
