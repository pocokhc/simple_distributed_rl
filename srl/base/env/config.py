import logging
import pprint
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from srl.base.env.processor import EnvProcessor
from srl.base.spaces.space import SpaceBase
from srl.utils.serialize import apply_dict_to_dataclass, dataclass_to_dict, get_modified_fields, load_dict, save_dict

if TYPE_CHECKING:
    import gym
    import gymnasium

    from srl.base.env.base import EnvBase
    from srl.base.env.env_run import EnvRun
    from srl.base.env.gym_user_wrapper import GymUserWrapper


logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """環境の定義

    環境は名前だけ又はEnvConfigで定義します。
    EnvConfigからは細かい設定を追加できます。

    >>> env_config = "Grid"
    >>> env_config = srl.EnvConfig("Grid", {"move_reward": -0.1})

    生成は以下のどちらでもできます。

    >>> env = srl.make_env(env_config)
    >>> env = env_config.make()

    """

    #: Specifies the environment id
    id: str = ""

    #: 環境生成時に渡す引数を指定します。
    #: これは登録されているパラメータより優先されます。
    kwargs: dict = field(default_factory=dict)

    # --- episode option
    #: 1エピソードの最大ステップ数(0以下で無効)
    max_episode_steps: int = -1
    #: 1エピソードの最大実行時間(秒)(0以下で無効)
    episode_timeout: float = -1
    #: 1stepあたり、環境内で余分に進めるstep数
    #: 例えばframeskip=3の場合、ユーザが1step実行すると、環境内では4frame進みます。
    frameskip: int = 0
    #: 1以上を指定するとそのstep数以内で、エピソード開始からランダムstep進めてからエピソード開始します。
    random_noop_max: int = 0

    # --- gym
    #: gym/gymnasiumの環境を生成する場合、この関数を使って生成します。
    gym_make_func: Optional[Callable[..., Union["gym.Env", "gymnasium.Env"]]] = None
    #: gym/gymnasiumの環境に割り込むためのクラス、フレームワークの代わりに変換する
    #: (pickle化できる必要があります)
    gym_wrapper: Optional["GymUserWrapper"] = None
    #: gymとgymnasium両方ある場合にgymを使います。（デフォルトはgymnasium）
    use_gym: bool = False

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
    font_size: int = 18

    # --- override
    #: envのaction_spaceを上書きします。
    override_action_space: Optional[SpaceBase] = None
    #: envのobservation_spaceを上書きします。
    override_observation_space: Optional[SpaceBase] = None
    #: Processorを使う場合、定義したProcessorのリスト
    processors: List[EnvProcessor] = field(default_factory=list)

    # --- other
    #: action/observationの値をエラーが出ないように可能な限り変換します。
    #: ※エラー終了の可能性は減りますが、値の変換等による予期しない動作を引き起こす可能性が高くなります
    enable_sanitize: bool = False
    #: action/observationの値を厳密にチェックし、おかしい場合は例外を出力します。
    #: enable_assertionが有効な場合は、enable_sanitizeは無効です。
    enable_assertion: bool = False
    #: display name
    display_name: str = ""

    def __post_init__(self):
        self.__name: Optional[str] = None

    @property
    def name(self) -> str:
        if self.__name is None:
            if self.display_name != "":
                self.__name = self.display_name
            else:
                from srl.base.env.registration import make_base

                # 中で_nameが設定される
                env = make_base(self)
                try:
                    logger.debug("close")
                    env.close()
                except Exception:
                    logger.error(traceback.format_exc())
                assert self.__name is not None
        return self.__name

    def set_name(self, env: "EnvBase"):
        if self.__name is not None:
            return
        if self.display_name != "":
            self.__name = self.display_name
        else:
            name = env.get_display_name()
            if name != "":
                self.__name = name
            else:
                self.__name = self.id

    def make(self) -> "EnvRun":
        """環境を生成します。 make_env(env_config) と同じ動作です。"""
        from srl.base.env.registration import make

        return make(self)

    @classmethod
    def load(cls, path_or_cfg_dict: Union[dict, Any, str]) -> "EnvConfig":
        return apply_dict_to_dataclass(cls(""), load_dict(path_or_cfg_dict))

    def save(self, path: str):
        save_dict(dataclass_to_dict(self), path)
        return self

    def apply_dict(self, cfg_dict: Union[dict, Any]) -> "EnvConfig":
        return apply_dict_to_dataclass(self, cfg_dict)

    def to_dict(self, to_print: bool = False) -> dict:
        return dataclass_to_dict(self, to_print=to_print)

    def copy(self) -> "EnvConfig":
        config = EnvConfig.load(dataclass_to_dict(self))
        config.__name = self.__name
        return config

    def summary(self, show_changed_only: bool = False):
        if show_changed_only:
            d = get_modified_fields(self)
        else:
            d = dataclass_to_dict(self, to_print=True)
        print("--- EnvConfig ---\n" + pprint.pformat(d))


def load_env(path_or_cfg_dict: Union[dict, Any, str]) -> EnvConfig:
    return EnvConfig.load(path_or_cfg_dict)
