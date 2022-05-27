import enum
from typing import Dict, List, Union

import numpy as np

# space type
DiscreteSpaceType = Union[int, List[int]]
ContinuousSpaceType = Union[float, List[float], np.ndarray]
SpaceType = Union[int, List[int], float, List[float], np.ndarray]

# action type
DiscreteAction = int
ContinuousAction = List[float]
EnvAction = Union[SpaceType, None]
RLAction = Union[int, List[float], float]
EnvInvalidAction = Union[int, List[int]]
RLInvalidAction = int

# other type
EnvObservation = SpaceType
RLObservation = np.ndarray
Info = Dict[str, Union[float, int]]


class EnvObservationType(enum.Enum):
    UNKNOWN = 0
    # 値
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続
    # 画像
    GRAY_2ch = enum.auto()  # (width, height)
    GRAY_3ch = enum.auto()  # (width, height, 1)
    COLOR = enum.auto()  # (width, height, ch)
    # その他
    SHAPE2 = enum.auto()  # (width, height)
    SHAPE3 = enum.auto()  # (n, width, height)


class RLObservationType(enum.Enum):
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RLActionType(enum.Enum):
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RenderType(enum.Enum):
    NONE = ""
    Terminal = "Terminal"
    GUI = "GUI"
    # Notebook = "Notebook"
    RGB_Array = "RGB"
