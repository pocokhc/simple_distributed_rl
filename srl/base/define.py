import enum
from typing import Dict, List, Tuple, Union

import numpy as np

# --- space type
# DiscreteType  : int, List[int]
# ContinuousType: float, List[float], NDarray[np.float32]
SpaceType = Union[int, List[int], float, List[float], np.ndarray]

# --- action type
EnvActionType = Union[SpaceType, None]
EnvInvalidActionType = Union[int, List[int], np.ndarray]
RLActionType = Union[int, List[float], float, np.ndarray]
RLInvalidActionType = Union[int, np.ndarray]  # discrete only


# --- obs type
EnvObservationType = SpaceType
RLObservationType = Union[List[int], np.ndarray]

# --- info type
InfoType = Dict[str, Union[float, int, str]]


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


class EnvObservationTypes(enum.Enum):
    UNKNOWN = 0
    # value
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()
    # image
    GRAY_2ch = enum.auto()  # (height, width)
    GRAY_3ch = enum.auto()  # (height, width, 1)
    COLOR = enum.auto()  # (height, width, 3)
    IMAGE = enum.auto()  # (height, width, ch)


class RLBaseTypes(enum.Enum):
    ANY = enum.auto()
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()


class RLTypes(enum.Enum):
    UNKNOWN = 0
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()  # np
    IMAGE = enum.auto()  # (height, width, ch)


class RLMemoryTypes(enum.Enum):
    NONE = enum.auto()
    SEQUENCE = enum.auto()
    BUFFER = enum.auto()
    PRIORITY = enum.auto()


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
