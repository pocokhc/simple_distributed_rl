import enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypeVar, Union

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
    Union[str, int, Tuple[Union[str, int], ...], List[Union[str, int]]],
    Union[EnvActionType, Tuple[int, EnvActionType]],
]

# --- workers
PlayerType = Union[
    None,
    str,  # name
    Tuple[str, dict],  # [name, kwargs]
    "RLConfig",  # RLConfig
    Tuple["RLConfig", Any],  # [RLConfig, RLParameter]
]

# --- Generic ---
TConfig = TypeVar("TConfig")
TParameter = TypeVar("TParameter")
TActSpace = TypeVar("TActSpace")
TActType = TypeVar("TActType")
TObsSpace = TypeVar("TObsSpace")
TObsType = TypeVar("TObsType")


class DoneTypes(enum.Enum):
    RESET = enum.auto()
    NONE = enum.auto()
    TERMINATED = enum.auto()
    TRUNCATED = enum.auto()

    @staticmethod
    def done(done: Union[bool, "DoneTypes"]) -> bool:
        return done if isinstance(done, bool) else DoneTypes.NONE != done

    @staticmethod
    def from_bool(done: Union[bool, "DoneTypes"]) -> "DoneTypes":
        if isinstance(done, bool):
            return DoneTypes.TERMINATED if done else DoneTypes.NONE
        return done


class SpaceTypes(enum.Enum):
    UNKNOWN = 0
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
    CONTINUOUS = enum.auto()
    IMAGE = enum.auto()
    TEXT = enum.auto()
    MULTI = enum.auto()


class RLMemoryTypes(enum.Enum):
    UNKNOWN = enum.auto()
    SEQUENCE = enum.auto()
    BUFFER = enum.auto()
    PRIORITY = enum.auto()


class ObservationModes(enum.Flag):
    ENV = enum.auto()
    RENDER_IMAGE = enum.auto()
    RENDER_TERMINAL = enum.auto()

    @staticmethod
    def from_str(mode: Union[str, "ObservationModes"]) -> "ObservationModes":
        if isinstance(mode, str):
            mode = mode.strip().lower()
            if mode == "":
                mode = ObservationModes.ENV
            elif mode in ["img", "image", "render", "render_img", "render_image"]:
                mode = ObservationModes.RENDER_IMAGE
            else:
                mode = ObservationModes.ENV
        return mode


class RenderModes(enum.Enum):
    none = 0
    terminal = enum.auto()
    ansi = enum.auto()  # hidden mode
    window = enum.auto()
    rgb_array = enum.auto()  # hidden mode

    @staticmethod
    def get_names() -> List[str]:
        return [i.name for i in RenderModes]

    @staticmethod
    def from_str(mode: Union[str, "RenderModes"]) -> "RenderModes":
        if isinstance(mode, str):
            if mode == "":
                mode = "none"
            names = RenderModes.get_names()
            assert mode in names, "Unknown mode '{}'. mode list is [{}].".format(
                mode,
                ",".join(names),
            )
            mode = RenderModes[mode]
        return mode

    @staticmethod
    def is_rendering(mode: Union[str, "RenderModes"]) -> bool:
        return not (mode == "" or mode == RenderModes.none)
