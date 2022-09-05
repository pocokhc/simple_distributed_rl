import itertools
import logging
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation
from srl.base.env.base import SpaceBase

logger = logging.getLogger(__name__)


class ArrayDiscreteSpace(SpaceBase[List[int]]):
    def __init__(
        self,
        size: int,
        low: Optional[Union[int, List[int]]] = None,
        high: Optional[Union[int, List[int]]] = None,
    ) -> None:
        self._size = size
        assert isinstance(size, int)

        if low is None:
            self._low = None
        else:
            self._low = [low for _ in range(self.size)] if isinstance(low, int) else low
            assert len(self._low) == size

        if high is None:
            self._high = None
        else:
            self._high = [high for _ in range(self.size)] if isinstance(high, int) else high
            assert len(self._high) == size

        self.decode_tbl = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def low(self) -> Optional[List[int]]:
        return self._low

    @property
    def high(self) -> Optional[List[int]]:
        return self._high

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> List[int]:
        assert self.low is not None and self.high is not None

        self._create_action_tbl()

        valid_actions = []
        for a in self.decode_tbl:  # decode_tbl is all action
            if a not in invalid_actions:
                valid_actions.append(a)

        return list(random.choice(valid_actions))

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != self.size:
            return False
        for i in range(self.size):
            if self.low is not None:
                if val[i] < self.low[i]:
                    return False
            if self.high is not None:
                if val[i] > self.high[i]:
                    return False
        return True

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

    def __str__(self) -> str:
        low = None if self.low is None else np.min(self.low)
        high = None if self.high is None else np.max(self.high)
        return f"ArrayDiscrete({self.size}|{low}, {high})"

    # --- discrete
    def _create_action_tbl(self) -> None:
        assert self.low is not None and self.high is not None
        if self.decode_tbl is not None:
            return

        if self.size > 10:
            logger.warning("It may take some time.")

        arr_list = [[a for a in range(self.low[i], self.high[i])] for i in range(self.size)]

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
