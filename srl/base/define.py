import enum


class EnvActionType(enum.Enum):
    UNKOWN = 0
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class EnvObservationType(enum.Enum):
    UNKOWN = 0
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


class RLActionType(enum.Enum):
    UNKOWN = 0
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RLObservationType(enum.Enum):
    UNKOWN = 0
    ANY = enum.auto()
    DISCRETE = enum.auto()  # 離散
    CONTINUOUS = enum.auto()  # 連続


class RenderType(enum.Enum):
    NONE = 0
    Terminal = enum.auto()
    GUI = enum.auto()
    Notebook = enum.auto()
    RGB_Array = enum.auto()
