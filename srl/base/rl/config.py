import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import numpy as np

from srl.base.define import EnvObservationTypes, RLBaseTypes, RLObservationType, RLTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.exception import UndefinedError
from srl.base.rl.processor import Processor
from srl.base.spaces.box import BoxSpace
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker


logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC):
    """RLConfig はアルゴリズムの動作を定義します。
    アルゴリズム毎に別々のハイパーパラメータがありますが、ここはアルゴリズム共通のパラメータの定義となります。
    """

    #: env_observation_type を上書きできます。
    #: 例えばgymの自動判定で想定外のTypeになった場合、ここで上書きできます。
    override_env_observation_type: EnvObservationTypes = EnvObservationTypes.UNKNOWN
    #: action_type を上書きできます。
    #: これはアルゴリズム側の base_action_type がANY(Discrete/Continuousどちらも対応できるアルゴリズム)の場合のみ有効になります。
    override_action_type: RLTypes = RLTypes.UNKNOWN

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    #: The number of divisions when converting from continuous to discrete values.
    #: If -1, round by round transform.
    action_division_num: int = 5

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    #: The number of divisions when converting from continuous to discrete values.
    #: If -1, round by round transform.
    observation_division_num: int = -1

    #: ExtendWorkerを使う場合に指定
    extend_worker: Optional[Type["ExtendWorker"]] = None
    #: 指定されていた場合、Parameter生成時にpathファイルをロードします
    parameter_path: str = ""
    #: 指定されていた場合、Memory生成時にpathファイルをロードします
    memory_path: str = ""
    #: Trueの場合、アルゴリズム側で指定されたprocessorを使用します
    use_rl_processor: bool = True

    #: 状態の入力をrender_imageに変更します。環境からの状態は画像に上書きされます。
    #: Change state input to render_image. Existing settings will be overwritten.
    use_render_image_for_observation: bool = False

    #: Processorを使う場合、定義したProcessorのリスト
    processors: List[Processor] = field(default_factory=list)

    # --- Worker Config
    #: state_encodeを有効にするか
    enable_state_encode: bool = True
    #: action_decodeを有効にするか
    enable_action_decode: bool = True
    #: reward_encodeを有効にするか
    enable_reward_encode: bool = True
    #: done_encodeを有効にするか
    enable_done_encode: bool = True
    #: 過去Nステップをまとめて状態とします
    window_length: int = 1
    #: window_length指定時の存在しないstepでの状態の値
    dummy_state_val: float = 0.0

    #: memoryデータを圧縮してやり取りするかどうか
    memory_compress: bool = True

    # --- other
    #: action/observationの値をエラーが出ないように可能な限り変換します。
    #: ※エラー終了の可能性は減りますが、値の変換等による予期しない動作を引き起こす可能性が高くなります
    enable_sanitize_value: bool = True
    #: action/observationの値を厳密にチェックし、おかしい場合は例外を出力します。
    #: enable_assertion_valueが有効な場合は、enable_sanitize_valueは無効です。
    enable_assertion_value: bool = False

    def __post_init__(self) -> None:
        self._is_setup = False
        self._run_processors: List[Processor] = []
        self._rl_action_type = self.override_action_type
        self._input_is_image = False

        # The device used by the framework.
        self._used_device_tf: str = "/CPU"
        self._used_device_torch: str = "cpu"

        self._changeable_parameter_names_base = [
            "parameter_path",
            "memory_path",
            "enable_sanitize_value",
            "enable_assertion_value",
        ]
        self._changeable_parameter_names: List[str] = []
        self._check_parameter = True  # last

    def assert_params(self) -> None:
        assert self.window_length > 0

    # ----------------------------
    # RL config
    # ----------------------------
    @abstractmethod
    def getName(self) -> str:  # get_nameに変えたい…
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_action_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_observation_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_use_framework(self) -> str:
        raise NotImplementedError()

    def set_config_by_env(self, env: EnvRun) -> None:
        pass  # NotImplemented

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def set_processor(self) -> List[Processor]:
        return []  # NotImplemented

    def get_changeable_parameters(self) -> List[str]:
        return []  # NotImplemented

    @property
    def use_backup_restore(self) -> bool:
        return False

    # infoの情報のタイプを指定、出力形式等で使用を想定
    # 各行の句は省略可能
    # name : {
    #   "type": 型を指定(None, int, float, str)
    #   "data": 以下のデータ形式を指定
    #   "ave" : 平均値を使用(default)
    #   "last": 最後のデータを使用
    #   "min" : 最小値
    #   "max" : 最大値
    # }
    @property
    def info_types(self) -> dict:
        return {}  # NotImplemented

    # ----------------------------
    # setup
    # ----------------------------
    def setup(self, env: EnvRun, enable_log: bool = True) -> None:
        if self._is_setup:
            return
        self._check_parameter = False

        if enable_log:
            logger.info(f"--- {self.getName()}")
            logger.info(f"max_episode_steps     : {env.max_episode_steps}")
            logger.info(f"player_num            : {env.player_num}")
            logger.info(f"observation_type(env) : {env.observation_type}")
            logger.info(f"observation_space(env): {env.observation_space}")

        # env property
        self.env_max_episode_steps = env.max_episode_steps
        self.env_player_num = env.player_num

        self._env_action_space = env.action_space  # action_spaceはenvを使いまわす
        rl_observation_space = env.observation_space
        rl_env_observation_type = env.observation_type

        # --- backup/restore check
        if self.use_backup_restore:
            try:
                d = env.backup()
                env.restore(d)
            except Exception:
                logger.error(f"'{self.getName()}' uses restore/backup, but it is not implemented in {env.name}.")
                raise

        # -----------------------
        # observation
        # -----------------------

        # --- observation_typeの上書き
        if self.override_env_observation_type != EnvObservationTypes.UNKNOWN:
            rl_env_observation_type = self.override_env_observation_type
            if enable_log:
                logger.info(f"override observation type: {rl_env_observation_type}")

        # --- processor
        self._run_processors = []
        if self.enable_state_encode:
            # render image
            if self.use_render_image_for_observation:
                from srl.rl.processors.render_image_processor import RenderImageProcessor

                self._run_processors.append(RenderImageProcessor())

        # user processor
        self._run_processors.extend(self.processors)

        # rl processor
        if self.use_rl_processor:
            self._run_processors.extend(self.set_processor())

        if self.enable_state_encode:
            # change space
            for p in self._run_processors:
                _new_rl_observation_space, _new_rl_env_observation_type = p.preprocess_observation_space(
                    rl_observation_space,
                    rl_env_observation_type,
                    env,
                    self,
                )
                if enable_log:
                    logger.info(f"apply observation processor: {repr(p)}")
                    logger.info(f"  type : {rl_env_observation_type} -> {_new_rl_env_observation_type}")
                    logger.info(f"  space: {rl_observation_space} -> {_new_rl_observation_space}")
                rl_observation_space = _new_rl_observation_space
                rl_env_observation_type = _new_rl_env_observation_type

        [r.setup(env, self) for r in self._run_processors]

        self._rl_observation_one_step_space = rl_observation_space
        self._rl_observation_space = rl_observation_space
        self._rl_env_observation_type = rl_env_observation_type

        # --- window_length
        if self.window_length > 1:
            self._rl_observation_space = BoxSpace(
                (self.window_length,) + self._rl_observation_one_step_space.shape,
                np.min(self._rl_observation_one_step_space.low),
                np.max(self._rl_observation_one_step_space.high),
            )

        # --- obs type
        # 優先度
        # 1. RL
        # 2. obs_space
        if self.base_observation_type == RLBaseTypes.DISCRETE:
            self._rl_observation_type: RLTypes = RLTypes.DISCRETE
        elif self.base_observation_type == RLBaseTypes.CONTINUOUS:
            self._rl_observation_type: RLTypes = RLTypes.CONTINUOUS
        elif self.base_observation_type == RLBaseTypes.ANY:
            self._rl_observation_type: RLTypes = self._rl_observation_space.rl_type
        else:
            raise UndefinedError(self.base_observation_type)

        # CONTINUOUSなら画像チェックする
        if self._rl_observation_type == RLTypes.CONTINUOUS:
            if self._rl_env_observation_type in [
                EnvObservationTypes.GRAY_2ch,
                EnvObservationTypes.GRAY_3ch,
                EnvObservationTypes.COLOR,
                EnvObservationTypes.IMAGE,
            ]:
                self._rl_observation_type = RLTypes.IMAGE

        # -----------------------
        #  action type
        # -----------------------
        # 優先度
        # 1. RL
        # 2. override_action_type
        # 3. action_space
        if self.base_action_type == RLBaseTypes.DISCRETE:
            self._rl_action_type: RLTypes = RLTypes.DISCRETE
        elif self.base_action_type == RLBaseTypes.CONTINUOUS:
            self._rl_action_type: RLTypes = RLTypes.CONTINUOUS
        elif self.base_action_type == RLBaseTypes.ANY:
            if self.override_action_type != RLTypes.UNKNOWN:
                self._rl_action_type: RLTypes = self.override_action_type
            else:
                self._rl_action_type: RLTypes = self._env_action_space.rl_type
        else:
            raise UndefinedError(self.base_action_type)

        # ------------------------------

        # --- division
        # RLが DISCRETE で Space が CONTINUOUS なら分割して DISCRETE にする
        if (self._rl_action_type == RLTypes.DISCRETE) and (self._env_action_space.rl_type == RLTypes.CONTINUOUS):
            self._env_action_space.create_division_tbl(self.action_division_num)
        if (self._rl_observation_type == RLTypes.DISCRETE) and (
            self._rl_observation_space.rl_type == RLTypes.CONTINUOUS
        ):
            self._rl_observation_space.create_division_tbl(self.observation_division_num)

        # --- set rl property
        if self._rl_action_type == RLTypes.DISCRETE:
            self._action_num = self.action_space.n
            self._action_low = np.ndarray(0)
            self._action_high = np.ndarray(self._action_num - 1)
        else:
            # ANYの場合もCONTINUOUS
            self._action_num = self.action_space.list_size
            self._action_low = np.array(self.action_space.list_low)
            self._action_high = np.array(self.action_space.list_high)

        self._input_is_image = self._rl_env_observation_type not in [
            EnvObservationTypes.DISCRETE,
            EnvObservationTypes.CONTINUOUS,
            EnvObservationTypes.UNKNOWN,
        ]

        # --- option
        self.set_config_by_env(env)

        # --- changeable parameters
        self._changeable_parameter_names = self._changeable_parameter_names_base[:]
        self._changeable_parameter_names.extend(self.get_changeable_parameters())

        # --- log
        self._check_parameter = True
        self._is_setup = True
        if enable_log:
            logger.info(f"action_type(rl)               : {self._rl_action_type}")
            logger.info(f"action_space(env)             : {self._env_action_space}")
            logger.info(f"observation_env_type(rl)      : {self._rl_env_observation_type}")
            logger.info(f"observation_type(rl)          : {self._rl_observation_type}")
            logger.info(f"observation_space(rl)         : {self._rl_observation_space}")
            if self.window_length > 1:
                logger.info(f"observation_one_step_space(rl): {self._rl_observation_one_step_space}")

    def __setattr__(self, name: str, value):
        if name == "_is_setup":
            object.__setattr__(self, name, value)
            return

        # --- パラメータが決まった後の書き換え
        if getattr(self, "_check_parameter", False) and (not name.startswith("_")):
            if not hasattr(self, name):
                logger.warning(f"An undefined variable was assigned. {name}={value}")
            elif getattr(self, "_is_setup", False):
                if name not in getattr(self, "_changeable_parameter_names", []):
                    s = f"Parameter has been rewritten. '{name}' : '{getattr(self,name)}' -> '{value}'"
                    logger.info(s)

        object.__setattr__(self, name, value)

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def name(self) -> str:
        return self.getName()

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    @property
    def run_processors(self) -> List[Processor]:
        return self._run_processors

    @property
    def used_device_tf(self) -> str:
        return self._used_device_tf

    @property
    def used_device_torch(self) -> str:
        return self._used_device_torch

    @property
    def action_space(self) -> SpaceBase:
        return self._env_action_space

    @property
    def action_type(self) -> RLTypes:
        return self._rl_action_type

    @property
    def observation_one_step_space(self) -> SpaceBase:
        return self._rl_observation_one_step_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._rl_observation_space

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._rl_observation_space.shape

    @property
    def observation_type(self) -> RLTypes:
        return self._rl_observation_type

    @property
    def env_observation_type(self) -> EnvObservationTypes:
        return self._rl_env_observation_type

    @property
    def input_is_image(self) -> bool:
        return self._input_is_image

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self, reset_env_config: bool = False) -> Any:
        config = self.__class__()
        config._check_parameter = False

        for k, v in self.__dict__.items():
            if k == "_check_parameter":
                continue
            if isinstance(v, EnvRun):
                continue
            try:
                setattr(config, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        if reset_env_config:
            config._is_setup = False
        else:
            config._is_setup = self._is_setup
        return config

    def create_dummy_state(self, is_one: bool = False) -> RLObservationType:
        if is_one:
            return np.full(self._rl_observation_one_step_space.shape, self.dummy_state_val, dtype=np.float32)
        else:
            return np.full(self.observation_shape, self.dummy_state_val, dtype=np.float32)

    # ----------------------------------
    # rl use property(reset後に使えます)
    # ----------------------------------
    @property
    def action_num(self) -> int:
        return self._action_num

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high


@dataclass
class DummyRLConfig(RLConfig):
    name: str = "dummy"

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    def get_use_framework(self) -> str:
        return ""

    def getName(self) -> str:
        return self.name
