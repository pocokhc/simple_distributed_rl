import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    import gym
    import gymnasium

    from srl.base.env.base import EnvBase
    from srl.base.env.env_run import EnvRun
    from srl.base.env.gym_user_wrapper import GymUserWrapper


logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """EnvConfig は環境を定義します。

    環境は、基本は名前だけで呼び出せます。

    >>> env = srl.make_env("Grid")

    しかし、ユーザが環境のパラメータを変えたい場合、このクラスでパラメータを変更します。

    >>> env_config = srl.EnvConfig("Grid")
    >>> env = srl.make_env(env_config)

    """

    #: Specifies the environment name
    name: str

    #: 環境生成時に渡す引数を指定します。
    #: これは登録されているパラメータより優先されます。
    kwargs: Dict = field(default_factory=dict)

    # --- episode option
    #: 1エピソードの最大ステップ数(0以下で無効)
    max_episode_steps: int = -1
    #: 1エピソードの最大実行時間(秒)(0以下で無効)
    episode_timeout: float = -1
    #: 1stepあたり、環境内で余分に進めるstep数
    #: 例えばframeskip=3の場合、ユーザが1step実行すると、環境内では4frame進みます。
    frameskip: int = 0

    # --- gym
    #: gymの環境を生成する場合、Noneの場合は "gym.make" で生成されますが、関数を渡すとその関数で生成されます。
    gym_make_func: Optional[Callable[..., "gym.Env"]] = None
    #: gymnasiumの環境を生成する場合、Noneの場合は "gymnasium.make" で生成されますが、関数を渡すとその関数で生成されます。
    gymnasium_make_func: Optional[Callable[..., "gymnasium.Env"]] = None
    #: gym/gymnasium環境を生成する際に、observation_space が画像かどうかチェックします
    gym_check_image: bool = True
    #: gym/gymnasium環境を生成する際に space の確認を実際にシミュレートして確認する
    gym_prediction_by_simulation: bool = True
    #: gym_prediction_by_simulation のシミュレーションステップ数
    gym_prediction_step: int = 10
    #: gym/gymnasiumの環境に割り込むためのクラス
    #: (pickle化できる必要があります)
    gym_wrappers: List["GymUserWrapper"] = field(default_factory=list)

    # --- render option
    #: renderの間隔(ms)
    #: - -1 の場合、Envで定義されている値が使われます。
    render_interval: float = -1
    #: renderのscale
    render_scale: float = 1.0
    #: render時のフォント名
    #: ""の場合は、`srl.font.PlemolJPConsoleHS-Regular.ttf` が使われます。
    font_name: str = ""
    #: render時のフォントサイズ
    font_size: int = 12

    # --- other
    #: action/observationの値をエラーが出ないように可能な限り変換します。
    #: ※エラー終了の可能性は減りますが、値の変換等による予期しない動作を引き起こす可能性が高くなります
    enable_sanitize_value: bool = True
    #: action/observationの値を厳密にチェックし、おかしい場合は例外を出力します。
    #: enable_assertion_valueが有効な場合は、enable_sanitize_valueは無効です。
    enable_assertion_value: bool = False

    def make_env(self) -> "EnvRun":
        """環境を生成します。 srl.make_env(env_config) と同じ動作です。"""
        from srl.base.env.registration import make

        return make(self)

    def _update_env_info(self, env: "EnvBase"):
        """env 作成時に env 情報を元に Config を更新"""
        if self.max_episode_steps <= 0:
            self.max_episode_steps = env.max_episode_steps
        self.player_num = env.player_num

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self) -> "EnvConfig":
        config = EnvConfig(self.name)
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
            elif type(v) in [list, dict]:
                setattr(config, k, copy.deepcopy(v))
        return config
