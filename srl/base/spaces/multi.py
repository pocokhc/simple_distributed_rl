import logging
import random
import time
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError

from .space import SpaceBase

logger = logging.getLogger(__name__)


class MultiSpace(SpaceBase[list]):
    def __init__(self, spaces: List[SpaceBase]) -> None:
        self.spaces = spaces
        self.decode_tbl = None
        self.encode_tbl = None

        self._is_discrete = True
        for s in self.spaces:
            if s.stype != SpaceTypes.DISCRETE:
                self._is_discrete = False
                break

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def space_size(self) -> int:
        return len(self.spaces)

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.MULTI

    @property
    def dtype(self):
        raise NotSupportedError()

    def sample(self, mask: List[list] = []) -> list:
        if self._is_discrete:
            if len(mask) > 0:
                acts = self.get_valid_actions(mask)
                return random.choice(acts)
            else:
                return [s.sample() for s in self.spaces]
        else:
            if len(mask) > 0:
                logger.info(f"mask is not support: {mask}")
            return [s.sample() for s in self.spaces]

    def get_valid_actions(self, masks: List[list] = []) -> list:
        if not self._is_discrete:
            raise NotSupportedError()

        import itertools

        valid_acts = [s.get_valid_actions() for s in self.spaces]
        valid_acts = list(itertools.product(*valid_acts))
        if len(masks) == 0:
            return valid_acts

        # maskを除外
        valid_acts2 = []
        for acts in valid_acts:
            in_mask = False
            for mask in masks:
                is_mask = True
                for i in range(len(acts)):
                    if isinstance(acts[i], np.ndarray):
                        if not (acts[i] == mask[i]).all():
                            is_mask = False
                            break
                    else:
                        if acts[i] != mask[i]:
                            is_mask = False
                            break
                if is_mask:
                    in_mask = True
                    break
            if not in_mask:
                valid_acts2.append(acts)

        return valid_acts2

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

    def to_str(self, val: list) -> str:
        return "_".join([s.to_str(v) for v, s in zip(val, self.spaces)])

    def get_default(self) -> list:
        return [s.get_default() for s in self.spaces]

    def copy(self) -> "MultiSpace":
        o = MultiSpace([s.copy() for s in self.spaces])
        o.decode_tbl = self.decode_tbl
        o.encode_tbl = self.encode_tbl
        return o

    def copy_value(self, val: list) -> list:
        return [s.copy_value(v) for v, s in zip(val, self.spaces)]

    def __eq__(self, o: "MultiSpace") -> bool:
        if not isinstance(o, MultiSpace):
            return False
        if len(self.spaces) != len(o.spaces):
            return False
        for i in range(len(self.spaces)):
            if self.spaces[i] != o.spaces[i]:
                return False
        return True

    def __str__(self) -> str:
        s = f"MultiSpace({len(self.spaces)})"
        for p in self.spaces:
            s += f"\n {str(p)}"
        return s

    def create_division_tbl(self, division_num: int) -> None:
        [s.create_division_tbl(division_num) for s in self.spaces]

    # --- stack
    def create_stack_space(self, length: int):
        spaces = [_s.create_stack_space(length) for _s in self.spaces]
        return MultiSpace(spaces)

    def encode_stack(self, val: list):
        return [self.spaces[i].encode_stack([v[i] for v in val]) for i in range(self.space_size)]

    # --------------------------------------
    # create_tbl
    # --------------------------------------
    def _create_tbl(self) -> None:
        if self.decode_tbl is not None:
            return
        import itertools

        t0 = time.time()
        arr_list = [[a for a in range(s.int_size)] for s in self.spaces]
        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i
        logger.info(f"create table time: {time.time() - t0:.1f}s")

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        self._create_tbl()
        assert self.decode_tbl is not None
        return len(self.decode_tbl)

    def encode_to_int(self, val: list) -> int:
        self._create_tbl()
        assert self.encode_tbl is not None
        key = [s.encode_to_int(val[i]) for i, s in enumerate(self.spaces)]
        return self.encode_tbl[tuple(key)]

    def decode_from_int(self, val: int) -> list:
        self._create_tbl()
        assert self.decode_tbl is not None
        vals = self.decode_tbl[val]
        return [s.decode_from_int(vals[i]) for i, s in enumerate(self.spaces)]

    # --------------------------------------
    # list int
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        return sum([s.list_int_size for s in self.spaces])

    @property
    def list_int_low(self) -> List[int]:
        return [x for space in self.spaces for x in space.list_int_low]

    @property
    def list_int_high(self) -> List[int]:
        return [x for space in self.spaces for x in space.list_int_high]

    def encode_to_list_int(self, val: list) -> List[int]:
        return [x for i, space in enumerate(self.spaces) for x in space.encode_to_list_int(val[i])]

    def decode_from_list_int(self, val: List[int]) -> list:
        arr = []
        n = 0
        for s in self.spaces:
            val2 = val[n : n + s.list_int_size]
            arr.append(s.decode_from_list_int(val2))
            n += s.list_int_size
        return arr

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return sum([s.list_float_size for s in self.spaces])

    @property
    def list_float_low(self) -> List[float]:
        return [x for space in self.spaces for x in space.list_float_low]

    @property
    def list_float_high(self) -> List[float]:
        return [x for space in self.spaces for x in space.list_float_high]

    def encode_to_list_float(self, val: list) -> List[float]:
        return [x for i, space in enumerate(self.spaces) for x in space.encode_to_list_float(val[i])]

    def decode_from_list_float(self, val: List[float]) -> list:
        arr = []
        n = 0
        for s in self.spaces:
            val2 = val[n : n + s.list_float_size]
            arr.append(s.decode_from_list_float(val2))
            n += s.list_float_size
        return arr

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (self.list_float_size,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array(self.list_float_low)

    @property
    def np_high(self) -> np.ndarray:
        return np.array(self.list_float_high)

    def encode_to_np(self, val: list, dtype) -> np.ndarray:
        return np.array(self.encode_to_list_float(val), dtype)

    def decode_from_np(self, val: np.ndarray) -> list:
        return self.decode_from_list_float(val.tolist())
