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


class RenderModes(enum.Enum):
    """IRender の set_render_mode でのみ使用"""

    NONE = 0
    Terminal = enum.auto()
    RBG_array = enum.auto()


class PlayRenderModes(enum.Enum):
    none = 0
    terminal = enum.auto()
    ansi = enum.auto()
    window = enum.auto()
    rgb_array = enum.auto()

    @staticmethod
    def get_names() -> List[str]:
        return [i.name for i in PlayRenderModes]

    @staticmethod
    def from_str(mode: Union[str, "PlayRenderModes"]) -> "PlayRenderModes":
        if isinstance(mode, str):
            if mode == "":
                mode = "none"
            names = PlayRenderModes.get_names()
            assert mode in names, "Unknown mode '{}'. mode list is [{}].".format(
                mode,
                ",".join(names),
            )
            mode = PlayRenderModes[mode]
        return mode

    @staticmethod
    def convert_render_mode(mode: "PlayRenderModes") -> RenderModes:
        return {
            PlayRenderModes.none: RenderModes.NONE,
            PlayRenderModes.terminal: RenderModes.Terminal,
            PlayRenderModes.ansi: RenderModes.Terminal,
            PlayRenderModes.window: RenderModes.RBG_array,
            PlayRenderModes.rgb_array: RenderModes.RBG_array,
        }[mode]
