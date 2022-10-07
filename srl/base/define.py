import enum
from typing import Dict, List, Optional, Tuple, Union

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
KeyBindType = Optional[Dict[Union[Tuple[Union[str, int], ...], str, int], EnvAction]]

# other type
EnvObservation = SpaceType
RLObservation = np.ndarray
Info = Dict[str, Union[float, int, str]]


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

    @staticmethod
    def is_image(val: "EnvObservationType") -> bool:
        return val in [
            EnvObservationType.GRAY_2ch,
            EnvObservationType.GRAY_3ch,
            EnvObservationType.COLOR,
        ]


class RLObservationType(enum.Enum):
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RLActionType(enum.Enum):
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RenderMode(enum.Enum):
    NONE = enum.auto()
    Terminal = enum.auto()
    RBG_array = enum.auto()


class PlayRenderMode(enum.Enum):
    none = enum.auto()
    terminal = enum.auto()
    ansi = enum.auto()
    window = enum.auto()
    rgb_array = enum.auto()

    @staticmethod
    def get_names() -> List[str]:
        return [i.name for i in PlayRenderMode]

    @staticmethod
    def from_str(mode: Union[str, "PlayRenderMode"]) -> "PlayRenderMode":
        if isinstance(mode, str):
            if mode == "":
                mode = "none"
            names = PlayRenderMode.get_names()
            assert mode in names, "Unknown mode '{}'. mode list is [{}].".format(
                mode,
                ",".join(names),
            )
            mode = PlayRenderMode[mode]
        return mode

    @staticmethod
    def convert_render_mode(mode: "PlayRenderMode") -> RenderMode:
        return {
            PlayRenderMode.none: RenderMode.NONE,
            PlayRenderMode.terminal: RenderMode.Terminal,
            PlayRenderMode.ansi: RenderMode.Terminal,
            PlayRenderMode.window: RenderMode.RBG_array,
            PlayRenderMode.rgb_array: RenderMode.RBG_array,
        }[mode]
