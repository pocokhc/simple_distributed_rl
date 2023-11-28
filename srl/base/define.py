import enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# --- space type
# DiscreteType  : int, List[int]
# ContinuousType: float, List[float], NDarray[np.float32]
SpaceType = Union[int, List[int], float, List[float], np.ndarray]

# --- action type
EnvActionType = Union[SpaceType, None]
RLActionType = Union[int, List[float], float]
KeyBindType = Optional[Dict[Union[str, int, Tuple[Union[str, int], ...]], EnvActionType]]

# --- InvalidActionType
# int型の場合のみ対応（拡張はTODO）
InvalidActionType = int
InvalidActionsType = List[int]

# --- obs type
EnvObservationType = SpaceType
RLObservationType = np.ndarray

# --- info type
InfoType = Dict[str, Union[float, int, str]]


class EnvObservationTypes(enum.Enum):
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

    @staticmethod
    def is_image(val: "EnvObservationTypes") -> bool:
        return val in [
            EnvObservationTypes.GRAY_2ch,
            EnvObservationTypes.GRAY_3ch,
            EnvObservationTypes.COLOR,
        ]


class RLTypes(enum.Enum):
    ANY = 0
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


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
