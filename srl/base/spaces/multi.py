import logging
import random
import time
from typing import Any, List

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class MultiSpace(SpaceBase[list]):
    def __init__(self, spaces: List[SpaceBase]) -> None:
        super().__init__()
        self.spaces = spaces
        self.decode_tbl = None
        self.encode_tbl = None

        self._is_discrete = True
        for s in self.spaces:
            if s.stype != SpaceTypes.DISCRETE:
                self._is_discrete = False
                break

    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def space_size(self) -> int:
        return len(self.spaces)

    # ----------------------------------------

    @property
    def name(self) -> str:
        return "Multi"

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

    def equal_val(self, v1: list, v2: list) -> bool:
        if len(v1) != len(v2):
            return False
        for s, a, b in zip(self.spaces, v1, v2):
            if not s.equal_val(a, b):
                return False
        return True

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

    def create_division_tbl(
        self,
        division_num: int,
        max_size: int = 100_000,
        max_byte: int = 1024 * 1024 * 1024,
    ) -> None:
        [s.create_division_tbl(division_num, max_size, max_byte) for s in self.spaces]

    # --- stack
    def create_stack_space(self, length: int):
        spaces = [_s.create_stack_space(length) for _s in self.spaces]
        return MultiSpace(spaces)

    def encode_stack(self, val: list):
        return [self.spaces[i].encode_stack([v[i] for v in val]) for i in range(self.space_size)]

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_list(self):
        return [
            RLBaseTypes.MULTI,
            RLBaseTypes.BOX,
            RLBaseTypes.BOX_UNTYPED,
            RLBaseTypes.NP_ARRAY,
            RLBaseTypes.NP_ARRAY_UNTYPED,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.ARRAY_DISCRETE,
            RLBaseTypes.DISCRETE,
            RLBaseTypes.TEXT,
            # RLBaseTypes.CONTINUOUS, NG
        ]

    # --- DiscreteSpace
    def _create_tbl(self) -> None:
        if self.decode_tbl is not None:
            return
        import itertools

        t0 = time.time()
        disc_space_list = [s.create_encode_space_DiscreteSpace() for s in self.spaces]
        arr_list = [[a for a in range(s.n)] for s in disc_space_list]
        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i
        logger.info(f"create table time: {time.time() - t0:.1f}s")

    def create_encode_space_DiscreteSpace(self):
        from srl.base.spaces.discrete import DiscreteSpace

        self._create_tbl()
        assert self.decode_tbl is not None
        return DiscreteSpace(len(self.decode_tbl))  # startは0

    def encode_to_space_DiscreteSpace(self, val: list) -> int:
        self._create_tbl()
        assert self.encode_tbl is not None
        key = [s.encode_to_space_DiscreteSpace(val[i]) for i, s in enumerate(self.spaces)]
        return self.encode_tbl[tuple(key)]

    def decode_from_space_DiscreteSpace(self, val: int) -> list:
        self._create_tbl()
        assert self.decode_tbl is not None
        vals = self.decode_tbl[val]
        return [s.decode_from_space_DiscreteSpace(vals[i]) for i, s in enumerate(self.spaces)]

    # --- ArrayDiscreteSpace
    def _setup_ArrayDiscreteSpace(self):
        if hasattr(self, "_array_disc_spaces"):
            return
        self._array_disc_spaces = [s.create_encode_space_ArrayDiscreteSpace() for s in self.spaces]

    def create_encode_space_ArrayDiscreteSpace(self):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        self._setup_ArrayDiscreteSpace()
        size = sum([s.size for s in self._array_disc_spaces])
        low = []
        high = []
        for s in self._array_disc_spaces:
            low += s.low
            high += s.high
        return ArrayDiscreteSpace(size, low, high)

    def encode_to_space_ArrayDiscreteSpace(self, val: list) -> List[int]:
        self._setup_ArrayDiscreteSpace()
        x = []
        for v, s in zip(val, self.spaces):
            x += s.encode_to_space_ArrayDiscreteSpace(v)
        return x

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> list:
        self._setup_ArrayDiscreteSpace()
        arr = []
        n = 0
        for s, s2 in zip(self.spaces, self._array_disc_spaces):
            v = val[n : n + s2.size]
            arr.append(s.decode_from_space_ArrayDiscreteSpace(v))
            n += s2.size
        return arr

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        raise NotSupportedError()

    def encode_to_space_ContinuousSpace(self, val: list) -> float:
        raise NotSupportedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> list:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def _setup_ArrayContinuousSpace(self):
        if hasattr(self, "_array_cont_spaces"):
            return
        self._array_cont_spaces = [s.create_encode_space_ArrayContinuousSpace() for s in self.spaces]

    def create_encode_space_ArrayContinuousSpace(self):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        self._setup_ArrayContinuousSpace()
        size = sum([s.size for s in self._array_cont_spaces])
        low = []
        high = []
        for s in self._array_cont_spaces:
            low += s.low
            high += s.high
        return ArrayContinuousSpace(size, low, high)

    def encode_to_space_ArrayContinuousSpace(self, val: list) -> List[float]:
        self._setup_ArrayContinuousSpace()
        x = []
        for v, s in zip(val, self.spaces):
            x += s.encode_to_space_ArrayContinuousSpace(v)
        return x

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> list:
        self._setup_ArrayContinuousSpace()
        arr = []
        n = 0
        for s, s2 in zip(self.spaces, self._array_cont_spaces):
            v = val[n : n + s2.size]
            arr.append(s.decode_from_space_ArrayContinuousSpace(v))
            n += s2.size
        return arr

    # --- NpArray
    def _setup_NpArraySpace(self, dtype):
        if hasattr(self, "_np_array_spaces"):
            return self._np_array_space
        from srl.base.spaces.np_array import NpArraySpace

        self._np_array_spaces = [s.create_encode_space_NpArraySpace(dtype) for s in self.spaces]
        size = sum([s.size for s in self._np_array_spaces])
        low = []
        high = []
        for s in self._np_array_spaces:
            low += s.low.tolist()
            high += s.high.tolist()
        self._np_array_space = NpArraySpace(size, low, high, dtype)
        return self._np_array_space

    def create_encode_space_NpArraySpace(self, dtype):
        return self._setup_NpArraySpace(dtype)

    def encode_to_space_NpArraySpace(self, val: list, dtype) -> np.ndarray:
        self._setup_NpArraySpace(dtype)
        x = []
        for v, s in zip(val, self.spaces):
            x.append(s.encode_to_space_NpArraySpace(v, dtype))
        return np.concatenate(x).astype(dtype)

    def decode_from_space_NpArraySpace(self, val: np.ndarray) -> list:
        arr = []
        n = 0
        for s, s2 in zip(self.spaces, self._np_array_spaces):
            v = val[n : n + s2.size]
            arr.append(s.decode_from_space_NpArraySpace(v.astype(s.dtype)))
            n += s2.size
        return arr

    # --- NpArrayUnTyped
    def _setup_NpArrayUnTyped(self):
        if hasattr(self, "_np_array_untyped_space"):
            return self._np_array_untyped_space
        from srl.base.spaces.np_array import NpArraySpace

        self._np_array_untyped_spaces = [s.create_encode_space_NpArrayUnTyped() for s in self.spaces]
        size = sum([s.size for s in self._np_array_untyped_spaces])
        low = []
        high = []
        stype = SpaceTypes.DISCRETE
        for s in self._np_array_untyped_spaces:
            low += s.low.tolist()
            high += s.high.tolist()
            if s.stype == SpaceTypes.CONTINUOUS:
                stype = SpaceTypes.CONTINUOUS
        dtype = np.int64 if stype == SpaceTypes.DISCRETE else np.float32
        self._np_array_untyped_space = NpArraySpace(size, low, high, dtype)
        return self._np_array_untyped_space

    def create_encode_space_NpArrayUnTyped(self):
        return self._setup_NpArrayUnTyped()

    def encode_to_space_NpArrayUnTyped(self, val: list) -> np.ndarray:
        self._setup_NpArrayUnTyped()
        x = []
        for v, s in zip(val, self.spaces):
            x.append(s.encode_to_space_NpArrayUnTyped(v))
        return np.concatenate(x)

    def decode_from_space_NpArrayUnTyped(self, val: np.ndarray) -> list:
        self._setup_NpArrayUnTyped()
        arr = []
        n = 0
        for s, s2 in zip(self.spaces, self._np_array_untyped_spaces):
            v = val[n : n + s2.size]
            arr.append(s.decode_from_space_NpArrayUnTyped(v.astype(s.dtype)))
            n += s2.size
        return arr

    # --- Box
    def create_encode_space_Box(self, dtype):
        from srl.base.spaces.box import BoxSpace

        # shapeが同じならboxで使う、shapeが違うならNpArrayと同じ処理
        box_spaces = [s.create_encode_space_Box(dtype) for s in self.spaces]
        self._is_box_spaces_same_shape = len(set([tuple(s.shape) for s in box_spaces])) == 1

        if self._is_box_spaces_same_shape:
            shape = (len(box_spaces),) + box_spaces[0].shape
            low = np.asarray([s.low for s in box_spaces])
            high = np.asarray([s.high for s in box_spaces])
            return BoxSpace(shape, low, high, dtype)
        else:
            self._box_space_list = [s.create_encode_space_NpArraySpace(dtype) for s in self.spaces]
            size = sum([s.size for s in self._box_space_list])
            low = []
            high = []
            for s in self._box_space_list:
                low += s.low.tolist()
                high += s.high.tolist()
            return BoxSpace((size,), low, high, dtype)

    def encode_to_space_Box(self, val: list, dtype) -> np.ndarray:
        if getattr(self, "_is_box_spaces_same_shape", True):
            x = [s.encode_to_space_Box(v, dtype) for v, s in zip(val, self.spaces)]
            return np.asarray(x).astype(dtype)
        else:
            x = []
            for v, s in zip(val, self.spaces):
                x.append(s.encode_to_space_NpArraySpace(v, dtype))
            return np.concatenate(x).astype(dtype)

    def decode_from_space_Box(self, val: np.ndarray) -> list:
        if getattr(self, "_is_box_spaces_same_shape", True):
            x = [
                self.spaces[i].decode_from_space_Box(val[i].astype(self.spaces[i].dtype))
                for i in range(len(self.spaces))  #
            ]
            return x
        else:
            arr = []
            n = 0
            for s, s2 in zip(self.spaces, self._box_space_list):
                v = val[n : n + s2.size]
                arr.append(s.decode_from_space_NpArraySpace(v.astype(s.dtype)))
                n += s2.size
            return arr

    # --- BoxUnTyped
    def create_encode_space_BoxUnTyped(self):
        from srl.base.spaces.box import BoxSpace

        # shapeが同じならboxで使う、shapeが違うならNpArrayと同じ処理
        box_spaces = [s.create_encode_space_BoxUnTyped() for s in self.spaces]
        self._is_box_untyped_spaces_same_shape = len(set([tuple(s.shape) for s in box_spaces])) == 1

        # stype: DISCRETE -> CONTINUOUS
        # dtype: np.uint -> np.int64 -> np.float32
        if self._is_box_untyped_spaces_same_shape:
            shape = (len(box_spaces),) + box_spaces[0].shape
            low = np.asarray([s.low for s in box_spaces])
            high = np.asarray([s.high for s in box_spaces])
            dtype = np.uint
            stype = SpaceTypes.DISCRETE
            for s in box_spaces:
                if s.stype == SpaceTypes.CONTINUOUS:
                    stype = SpaceTypes.CONTINUOUS
                if "float" in str(s.dtype):
                    dtype = s.dtype
                elif ("float" not in str(dtype)) and ("uint" not in str(s.dtype)) and ("int" in str(s.dtype)):
                    dtype = s.dtype
            return BoxSpace(shape, low, high, dtype, stype)
        else:
            self._box_untyped_space_list = [s.create_encode_space_NpArrayUnTyped() for s in self.spaces]
            size = sum([s.size for s in self._box_untyped_space_list])
            low = []
            high = []
            dtype = np.uint
            stype = SpaceTypes.DISCRETE
            for s in self._box_untyped_space_list:
                low += s.low.tolist()
                high += s.high.tolist()
                if s.stype == SpaceTypes.CONTINUOUS:
                    stype = SpaceTypes.CONTINUOUS
                if "float" in str(s.dtype):
                    dtype = s.dtype
                elif ("float" not in str(dtype)) and ("uint" not in str(s.dtype)) and ("int" in str(s.dtype)):
                    dtype = s.dtype
            return BoxSpace((size,), low, high, dtype, stype)

    def encode_to_space_BoxUnTyped(self, val: list) -> np.ndarray:
        if getattr(self, "_is_box_untyped_spaces_same_shape", True):
            x = [s.encode_to_space_BoxUnTyped(v) for v, s in zip(val, self.spaces)]
            return np.asarray(x)
        else:
            x = []
            for v, s in zip(val, self.spaces):
                x.append(s.encode_to_space_NpArrayUnTyped(v))
            return np.concatenate(x)

    def decode_from_space_BoxUnTyped(self, val: np.ndarray) -> list:
        if getattr(self, "_is_box_untyped_spaces_same_shape", True):
            x = [
                self.spaces[i].decode_from_space_BoxUnTyped(val[i].astype(self.spaces[i].dtype))
                for i in range(len(self.spaces))  #
            ]
            return x
        else:
            arr = []
            n = 0
            for s, s2 in zip(self.spaces, self._box_untyped_space_list):
                v = val[n : n + s2.size]
                arr.append(s.decode_from_space_NpArrayUnTyped(v.astype(s.dtype)))
                n += s2.size
            return arr

    # --- TextSpace
    def _setup_TextSpace(self):
        if hasattr(self, "_text_spaces"):
            return
        self._text_spaces = [s.create_encode_space_TextSpace() for s in self.spaces]

    def create_encode_space_TextSpace(self):
        from srl.base.spaces.text import TextSpace

        self._setup_TextSpace()

        max_len = 0
        for s in self._text_spaces:
            if s.max_length <= 0:
                max_len = -1
                break
            max_len += s.max_length

        min_len = min([s.min_length for s in self._text_spaces])
        charset = "".join([s.charset for s in self._text_spaces])
        charset = "".join(dict.fromkeys(charset + "_"))

        return TextSpace(max_len, min_len, charset)

    def encode_to_space_TextSpace(self, val: list) -> str:
        self._setup_TextSpace()
        return "_".join([s.encode_to_space_TextSpace(v) for v, s in zip(val, self.spaces)])

    def decode_from_space_TextSpace(self, val: str) -> list:
        self._setup_TextSpace()
        arr = []
        vals = val.split("_")
        for i, s in enumerate(self.spaces):
            arr.append(s.decode_from_space_TextSpace(vals[i]))
        return arr

    # --- Multi
    def create_encode_space_MultiSpace(self):
        return self.copy()

    def encode_to_space_MultiSpace(self, val: list) -> list:
        return val

    def decode_from_space_MultiSpace(self, val: list) -> list:
        return val
