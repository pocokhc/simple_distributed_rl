import logging
import random
import time
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import RLTypes

from .space import SpaceBase

logger = logging.getLogger(__name__)


class ArraySpace(SpaceBase[list]):
    def __init__(self, spaces: List[SpaceBase]) -> None:
        self.spaces = spaces
        self.decode_tbl = None

        # --- check type
        self._rl_type = RLTypes.DISCRETE
        for s in self.spaces:
            if s.rl_type == RLTypes.CONTINUOUS:
                self._rl_type = RLTypes.CONTINUOUS
                break

    def sample(self, mask: List[list] = []) -> list:
        if len(mask) == 0:
            return [s.sample() for s in self.spaces]
        else:
            # 方針、discreteで値を出してdecodeする
            en_mask = [self.encode_to_int(m) for m in mask]
            valid_acts = [a for a in range(self.n) if a not in en_mask]
            a = random.choice(valid_acts)
            return self.decode_from_int(a)

    def get_valid_actions(self, mask: List[list] = []) -> List[list]:
        en_mask = [self.encode_to_int(m) for m in mask]
        valid_acts = [a for a in range(self.n) if a not in en_mask]
        return [self.decode_from_int(a) for a in valid_acts]

    def sanitize(self, val: Any) -> list:
        if isinstance(val, list):
            val = [s.sanitize(val[i]) for i, s in enumerate(self.spaces)]
        elif isinstance(val, tuple):
            val = [s.sanitize(val[i]) for i, s in enumerate(self.spaces)]
        elif isinstance(val, np.ndarray):
            val = val.flatten().tolist()
            val = [s.sanitize(val[i]) for i, s in enumerate(self.spaces)]
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != len(self.spaces):
            return False
        for i, s in enumerate(self.spaces):
            if not s.check_val(val[i]):
                return False
        return True

    @property
    def rl_type(self) -> RLTypes:
        return self._rl_type

    def get_default(self) -> list:
        return [s.get_default() for s in self.spaces]

    def __eq__(self, o: "ArraySpace") -> bool:
        if len(self.spaces) != len(o.spaces):
            return False
        for i, s in enumerate(self.spaces):
            if s != o.spaces[i]:
                return False
        return True

    def __str__(self) -> str:
        s = ", ".join([str(s) for s in self.spaces])
        return f"Array[{s}]"

    # --------------------------------------
    # create_tbl
    # --------------------------------------
    def _create_tbl(self) -> None:
        if self.decode_tbl is not None:
            return
        import itertools

        t0 = time.time()
        arr_list = [[a for a in range(s.n)] for s in self.spaces]
        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i
        logger.info(f"create table time: {time.time() - t0:.1f}s")

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def n(self) -> int:
        self._create_tbl()
        assert self.decode_tbl is not None
        return len(self.decode_tbl)

    def encode_to_int(self, val: list) -> int:
        self._create_tbl()
        key = [s.encode_to_int(val[i]) for i, s in enumerate(self.spaces)]
        return self.encode_tbl[tuple(key)]

    def decode_from_int(self, val: int) -> list:
        self._create_tbl()
        assert self.decode_tbl is not None
        vals = self.decode_tbl[val]
        return [s.decode_from_int(vals[i]) for i, s in enumerate(self.spaces)]

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    def encode_to_list_int(self, val: list) -> List[int]:
        arr = []
        for i, s in enumerate(self.spaces):
            arr.extend(s.encode_to_list_int(val[i]))
        return arr

    def decode_from_list_int(self, val: List[int]) -> list:
        arr = []
        n = 0
        for s in self.spaces:
            val2 = val[n : n + s.list_size]
            arr.append(s.decode_from_list_int(val2))
            n += s.list_size
        return arr

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return sum([s.list_size for s in self.spaces])

    @property
    def list_low(self) -> List[float]:
        arr = []
        for s in self.spaces:
            arr.extend(s.list_low)
        return arr

    @property
    def list_high(self) -> List[float]:
        arr = []
        for s in self.spaces:
            arr.extend(s.list_high)
        return arr

    def encode_to_list_float(self, val: list) -> List[float]:
        arr = []
        for i, s in enumerate(self.spaces):
            arr.extend(s.encode_to_list_float(val[i]))
        return arr

    def decode_from_list_float(self, val: List[float]) -> list:
        arr = []
        n = 0
        for s in self.spaces:
            val2 = val[n : n + s.list_size]
            arr.append(s.decode_from_list_float(val2))
            n += s.list_size
        return arr

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:  # List[Tuple]
        return [s.shape for s in self.spaces]

    @property
    def low(self) -> np.ndarray:  # List[np.ndarray]
        return [s.low for s in self.spaces]

    @property
    def high(self) -> np.ndarray:  # List[np.ndarray]
        return [s.high for s in self.spaces]

    def encode_to_np(self, val: list, dtype) -> np.ndarray:  # List[np.ndarray]
        # shapeが違うのでnpに出来ない TODO
        return [s.encode_to_np(val[i], dtype) for i, s in enumerate(self.spaces)]

    def decode_from_np(self, val: np.ndarray) -> list:  # List[np.ndarray]
        return [s.decode_from_np(val[i]) for i, s in enumerate(self.spaces)]
