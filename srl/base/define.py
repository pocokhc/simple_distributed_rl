import enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from srl.base.rl.config import RLConfig

# --- space type
# DiscreteType  : int, List[int]
# ContinuousType: float, List[float], NDarray[np.float32]
SpaceType = Union[
    int,
    List[int],
    float,
    List[float],
    np.ndarray,
]

# --- action type
EnvActionType = Union[SpaceType, None]
RLActionType = Union[
    int,
    List[float],
    float,
    np.ndarray,
    List[SpaceType],
    List["RLActionType"],
]

# --- obs type
EnvObservationType = SpaceType
RLObservationType = Union[
    List[int],
    np.ndarray,
    List[np.ndarray],
    List["RLObservationType"],
]


# --- KyeBind
# ActionSpace == not Array
#   {match key list : action}
#
# ActionSpace == Array
#   {match key list : [index, action]}
#
KeyBindType = Dict[
    Union[
        str,
        int,
        Tuple[Union[str, int], ...],
        List[Union[str, int]],
    ],
    Union[
        Any,
        Tuple[int, Any],
    ],
]

# --- workers
#: multi player option
#:    マルチプレイヤーゲームでのプレイヤーを指定します。
#:    None                : use rl_config worker
#:    str                 : Registered RuleWorker
#:    Tuple[str, dict]    : Registered RuleWorker(Pass kwargs argument)
#:    RLConfig            : use RLConfig worker
#:    Tuple[RLConfig, Any]: use RLConfig worker(Parameter)
PlayerType = Union[
    None,
    str,  # name
    Tuple[str, dict],  # [name, kwargs]
    "RLConfig",  # RLConfig
    Tuple["RLConfig", Any],  # [RLConfig, RLParameter]
]
PlayersType = Sequence[PlayerType]


class DoneTypes(enum.Enum):
    RESET = enum.auto()
    NONE = enum.auto()
    TERMINATED = enum.auto()
    TRUNCATED = enum.auto()
    ABORT = enum.auto()

    @staticmethod
    def from_val(done: Union[bool, str, "DoneTypes"]) -> "DoneTypes":
        if isinstance(done, bool):
            return DoneTypes.NONE if not done else DoneTypes.TERMINATED
        elif isinstance(done, str):
            try:
                return DoneTypes[done.upper()]
            except KeyError:
                raise ValueError(f"Invalid DoneType string: {done}")
        elif isinstance(done, DoneTypes):
            return done
        else:
            raise ValueError(f"Unsupported type for DoneType conversion: {type(done)}")


class SpaceTypes(enum.Enum):
    UNKNOWN = enum.auto()
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()
    GRAY_2ch = enum.auto()  # (height, width)
    GRAY_3ch = enum.auto()  # (height, width, 1)
    COLOR = enum.auto()  # (height, width, 3)
    IMAGE = enum.auto()  # (height, width, ch)
    TEXT = enum.auto()
    MULTI = enum.auto()  # list

    @staticmethod
    def is_image(t: "SpaceTypes") -> bool:
        return t in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
            SpaceTypes.IMAGE,
        ]


class RLBaseTypes(enum.Flag):
    NONE = enum.auto()
    DISCRETE = enum.auto()
    ARRAY_DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()
    ARRAY_CONTINUOUS = enum.auto()
    NP_ARRAY = enum.auto()
    BOX = enum.auto()
    TEXT = enum.auto()
    MULTI = enum.auto()

    @staticmethod
    def from_str(mode: Union[str, "RLBaseTypes"]) -> "RLBaseTypes":
        if isinstance(mode, RLBaseTypes):
            return mode
        elif isinstance(mode, str):
            mode_upper = mode.upper()
            if mode_upper in RLBaseTypes.__members__:
                return RLBaseTypes[mode_upper]
            else:
                raise ValueError(f"Unknown mode: {mode}, members: {list(RLBaseTypes.__members__.keys())}")
        else:
            raise TypeError(f"mode must be a str or RLBaseTypes, not {type(mode).__name__}")

    @staticmethod
    def to_list(flag: "RLBaseTypes") -> List["RLBaseTypes"]:
        return [f for f in type(flag) if (flag & f == f)]


RenderModeType = Literal[
    "",
    "terminal",
    "rgb_array",
    "window",
    "terminal_rgb_array",
]
