import enum


class EnvObservationType(enum.Enum):
    UNKOWN = 0
    # 値
    DISCRETE = 1  # 離散
    CONTINUOUS = 2  # 連続
    # 画像
    GRAY_2ch = 3  # (width, height)
    GRAY_3ch = 4  # (width, height, 1)
    COLOR = 5  # (width, height, ch)
    # その他
    SHAPE2 = 6  # (width, height)
    SHAPE3 = 7  # (n, width, height)


class RLActionType(enum.Enum):
    UNKOWN = 0
    DISCRETE = 1  # 離散
    CONTINUOUS = 2  # 連続


class RLObservationType(enum.Enum):
    UNKOWN = 0
    DISCRETE = 1  # 離散
    CONTINUOUS = 2  # 連続
