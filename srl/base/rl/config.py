import logging
import pickle
import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    PlayersType,
    RenderModeType,
    RLActionType,
    RLBaseTypes,
    RLObservationType,
    SpaceTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.exception import NotSupportedError, UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase, TActSpace, TObsSpace
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker
    from srl.base.rl.memory import RLMemory
    from srl.base.rl.parameter import RLParameter
    from srl.base.rl.processor import RLProcessor
    from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


TRLConfig = TypeVar("TRLConfig", bound="RLConfig", covariant=True)


@dataclass
class RLConfig(ABC, Generic[TActSpace, TObsSpace]):
    #: 状態の入力を指定
    observation_mode: Literal["", "render_image"] = ""

    #: env の observation_type を上書きします。
    #: 例えばgymの自動判定で想定外のTypeになった場合、ここで上書きできます。
    override_env_observation_type: SpaceTypes = SpaceTypes.UNKNOWN
    #: observation_type を上書きします。
    override_observation_type: Union[str, RLBaseTypes] = RLBaseTypes.NONE
    #: action_type を上書きします。
    override_action_type: Union[str, RLBaseTypes] = RLBaseTypes.NONE

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    action_division_num: int = 10

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    observation_division_num: int = 1000

    #: 1stepあたり、環境内で余分に進めるstep数
    #: 例えばframeskip=3の場合、1step実行すると、環境内では4frame進みます。
    frameskip: int = 0

    #: ExtendWorkerを使う場合に指定
    extend_worker: Optional[Type["ExtendWorker"]] = None
    #: Processorを使う場合に設定
    processors: List["RLProcessor"] = field(default_factory=list)
    #: render_image に対してProcessorを使う場合に設定(use_render_image_stateが有効なアルゴリズムの場合適用)
    render_image_processors: List["RLProcessor"] = field(default_factory=list)
    #: Trueの場合、アルゴリズム側で指定されたprocessorsを使用します
    enable_rl_processors: bool = True

    # --- Worker Config
    #: state_encodeを有効にするか
    enable_state_encode: bool = True
    #: action_decodeを有効にするか
    enable_action_decode: bool = True
    #: 2以上で過去Nステップをまとめて状態とします
    window_length: int = 1
    #: 2以上で過去Nステップをまとめて状態とします(use_render_image_stateが有効なアルゴリズムの場合適用)
    render_image_window_length: int = 1

    # --- render
    #: render時にエピソード終了時のstepで描画するか
    render_last_step: bool = True
    #: render時にRLへ入力される画像を描画するか
    render_rl_image: bool = True
    #: render時にRLへ入力される画像のサイズ
    render_rl_image_size: Tuple[int, int] = (128, 128)

    # --- other
    #: action/observationの値をエラーが出ないように可能な限り変換します。
    #: ※エラー終了の可能性は減りますが、値の変換等による予期しない動作を引き起こす可能性が高くなります
    enable_sanitize: bool = True
    #: action/observationの値を厳密にチェックし、おかしい場合は例外を出力します。
    #: enable_assertionが有効な場合は、enable_sanitizeは無効です。
    enable_assertion: bool = False
    #: dtype
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.__is_setup = False

        # The device used by the framework.
        self.__used_device_tf: str = "/CPU"
        self.__used_device_torch: str = "cpu"

        self.__applied_processors: List[RLProcessor] = []
        self.__applied_render_img_processors: List[RLProcessor] = []
        self.__request_env_render: RenderModeType = ""

        self.__changeable_parameter_names_base = [
            "parameter_path",
            "memory_path",
            "enable_sanitize",
            "enable_assertion",
        ]
        self.__changeable_parameter_names: List[str] = []
        self.__check_parameter = True  # last

    def get_dtype(self, framework: Literal["np", "numpy", "torch", "tf", "tensorflow"]) -> Any:
        if framework in ["np", "numpy"]:
            return getattr(np, self.dtype.lower())
        elif framework in ["tf", "tensorflow"]:
            import tensorflow as tf

            return getattr(tf, self.dtype.lower())
        elif framework in ["torch"]:
            import torch

            return getattr(torch, self.dtype.lower())
        else:
            return self.dtype

    def validate_params(self) -> None:
        if not (self.window_length > 0):
            raise ValueError(f"assert {self.window_length} > 0")
        if not (self.render_image_window_length > 0):
            raise ValueError(f"assert {self.render_image_window_length} > 0")

    # ----------------------------
    # RL config
    # ----------------------------
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_base_action_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_base_observation_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    def get_framework(self) -> str:
        return ""  # NotImplemented

    def get_processors(self, prev_observation_space: SpaceBase) -> List["RLProcessor"]:
        """前処理を追加したい場合設定"""
        return []  # NotImplemented

    def setup_from_env(self, env: EnvRun) -> None:
        """env初期化後に呼び出されます。env関係の初期化がある場合は記載してください。"""
        pass  # NotImplemented

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        """Actor関係の初期化がある場合は記載
        - 分散学習でactorが指定されたときに呼び出されます
        """
        pass  # NotImplemented

    def get_changeable_parameters(self) -> List[str]:
        # 設定が大変なので廃止の方向で…
        return []  # NotImplemented

    def use_backup_restore(self) -> bool:
        """envのbackup/restoreを使う場合はTrue, MCTSなどで使用"""
        return False  # NotImplemented

    def use_render_image_state(self) -> bool:
        """envの画像情報を使用
        - use_render_image_stateをTrueにするとworker.render_img_stateが有効になります
        - worker.render_img_state には env.render_rgb_array の画像が入ります
        """
        return False  # NotImplemented

    def get_render_image_processors(self, prev_observation_space: SpaceBase) -> List["RLProcessor"]:
        """render_img_stateに対する前処理" """
        return []  # NotImplemented

    def override_env_render_mode(self) -> RenderModeType:
        """envのrender_modeを上書き, humanのterminal入力等で使用"""
        return ""  # NotImplemented

    def use_update_parameter_from_worker(self) -> bool:
        """WorkerからParameterの更新がある場合はTrue
        - Trueの場合、分散学習で parameter.update_from_worker_parameter が学習後に呼ばれます
        - MCTSやGo系で使用
        """
        return False

    # ----------------------------
    # setup
    # ----------------------------
    def is_setup(self) -> bool:
        return self.__is_setup

    def setup(self, env: EnvRun, enable_log: bool = True) -> None:
        if self.__is_setup:
            return
        self.__check_parameter = False
        self.override_observation_type = RLBaseTypes.from_str(self.override_observation_type)
        self.override_action_type = RLBaseTypes.from_str(self.override_action_type)

        # --- metadata(表示用)
        logger.debug("--- Algorithm settings ---\n" + pprint.pformat(self.get_metadata()))

        # --- backup/restore check
        if self.use_backup_restore():
            try:
                env.setup()
                env.reset()
                d = env.backup()
                env.restore(d)
            except Exception:
                logger.error(f"'{self.get_name()}' uses restore/backup, but it is not implemented in {env.name}.")
                raise

        # -------------------------------------------------
        # observation space
        #  - env space(original) または observation_mode 後のspace(env_obs_space)
        #  - env_obs_spaceからprocessors後のspace(env_obs_space_in_rl)
        #  - env_obs_space_in_rlをRL側のbase_typeへ変換したspace(rl_obs_space_one_step)
        #  - window length==1: rl_obs_space = rl_obs_space_one_step
        #  - window length>1 : rl_obs_space_one_stepを window length したspace(rl_obs_space)
        # -------------------------------------------------

        # --- observation_mode による変更
        if self.observation_mode == "":
            env_obs_space = env.observation_space.copy()

            # --- observation_typeの上書き
            if isinstance(env_obs_space, BoxSpace):
                if self.override_env_observation_type != SpaceTypes.UNKNOWN:
                    if enable_log:
                        s = "override observation type: "
                        s += f"{env.observation_space} -> {self.override_env_observation_type}"
                        logger.info(s)
                    env_obs_space = env_obs_space.copy(stype=self.override_env_observation_type, is_stack_ch=None)
        elif self.observation_mode == "render_image":
            env.setup(render_mode="rgb_array")
            env.reset()
            rgb_array = env.render_rgb_array()
            if rgb_array is not None:
                self.__request_env_render = "rgb_array"
            else:
                rgb_array = env.render_terminal_text_to_image()
                if rgb_array is not None:
                    self.__request_env_render = "terminal"
                else:
                    raise NotSupportedError("Failed to get image.")
            env_obs_space = BoxSpace(rgb_array.shape, 0, 255, np.uint8, SpaceTypes.COLOR)
        else:
            raise UndefinedError(self.observation_mode)

        # --- apply processors
        self.__applied_processors = []
        self.__applied_render_img_processors = []
        self.__applied_remap_obs_processors: List[Tuple[Any, SpaceBase, SpaceBase]] = []
        self.__applied_remap_render_img_processors: List[Tuple[Any, SpaceBase, SpaceBase]] = []
        if self.enable_state_encode:
            # applied_processors list
            p_list: List[RLProcessor] = []
            if self.enable_rl_processors:
                p_list += self.get_processors(env_obs_space)  # algorithm processers
            p_list += self.processors  # user processors
            self.__applied_processors = [p.copy() for p in p_list]

            # remap
            for p in self.__applied_processors:
                prev_space = env_obs_space
                new_space = p.remap_observation_space(env_obs_space, env_run=env, rl_config=self)
                if new_space is not None:
                    env_obs_space = new_space
                    if enable_log and (prev_space != env_obs_space):
                        logger.info(f"apply obs processor: {repr(p)}")
                        logger.info(f"   {prev_space}")
                        logger.info(f" ->{env_obs_space}")
                    if hasattr(p, "remap_observation"):
                        self.__applied_remap_obs_processors.append((p, prev_space, env_obs_space))

        # one step はここで確定
        self.__env_obs_space_in_rl: SpaceBase = env_obs_space

        # --- obs type & space
        self.__rl_obs_space_one_step, self.__rl_obs_type = self._get_rl_space(
            required_type=self.get_base_observation_type(),
            override_type=self.override_observation_type,
            env_space=self.__env_obs_space_in_rl,
            division_num=self.observation_division_num,
        )

        # --- window_length
        if self.window_length > 1:
            self.__rl_obs_space = self.__rl_obs_space_one_step.create_stack_space(self.window_length)
        else:
            self.__rl_obs_space = self.__rl_obs_space_one_step

        # --- include render image
        if self.use_render_image_state():
            env.setup(render_mode="rgb_array")
            env.reset()
            rgb_array = env.render_rgb_array()
            if rgb_array is not None:
                self.__request_env_render = "rgb_array"
            else:
                rgb_array = env.render_terminal_text_to_image()
                if rgb_array is not None:
                    self.__request_env_render = "terminal"
                else:
                    raise NotSupportedError("Failed to get image.")
            self.__rl_obs_render_img_space_one_step: BoxSpace = BoxSpace(rgb_array.shape, 0, 255, np.uint8, SpaceTypes.COLOR)

            if self.enable_state_encode and self.enable_rl_processors:
                # applied_processors list
                self.__applied_render_img_processors = [p.copy() for p in self.get_render_image_processors(self.__rl_obs_render_img_space_one_step)]

                # remap
                for p in self.__applied_render_img_processors:
                    prev_space = self.__rl_obs_render_img_space_one_step
                    new_space = p.remap_observation_space(prev_space, env_run=env, rl_config=self)
                    if new_space is not None:
                        self.__rl_obs_render_img_space_one_step = cast(BoxSpace, new_space)
                        if enable_log and (prev_space != self.__rl_obs_render_img_space_one_step):
                            logger.info(f"apply img obs processor: {repr(p)}")
                            logger.info(f"   {prev_space}")
                            logger.info(f" ->{self.__rl_obs_render_img_space_one_step}")
                        if not isinstance(self.__rl_obs_render_img_space_one_step, BoxSpace):
                            logger.warning(f"render_image assumes BoxSpace. {self.__rl_obs_render_img_space_one_step=}")
                        if hasattr(p, "remap_observation"):
                            self.__applied_remap_render_img_processors.append((p, prev_space, self.__rl_obs_render_img_space_one_step))

            if self.render_image_window_length > 1:
                self.__rl_obs_render_img_space = self.__rl_obs_render_img_space_one_step.create_stack_space(self.render_image_window_length)
            else:
                self.__rl_obs_render_img_space = self.__rl_obs_render_img_space_one_step

        # -----------------------
        # action space, 2種類、特に前処理とかはないのでそのままenvと同じになる
        # 1. env space(original)
        # 2. rl action space
        # -----------------------
        self.__env_act_space = env.action_space.copy()

        # --- act type & space
        self.__rl_act_space, self.__rl_act_type = self._get_rl_space(
            required_type=self.get_base_action_type(),
            override_type=self.override_action_type,
            env_space=self.__env_act_space,
            division_num=self.action_division_num,
        )

        # --------------------------------------------

        # --- override_env_render_mode
        if self.override_env_render_mode() != "":
            if self.__request_env_render != "":
                logger.warning(f"'request_env_render' has been overridden: {self.__request_env_render} -> {self.override_env_render_mode()}")
            self.__request_env_render = self.override_env_render_mode()

        # --- validate
        self.validate_params()
        # validate_paramsを持ってる変数も実行
        for k, v in self.__dict__.items():
            if hasattr(v, "validate_params"):
                v.validate_params()

        # --- option
        self.setup_from_env(env)

        # --- changeable parameters
        self.__changeable_parameter_names = self.__changeable_parameter_names_base[:]
        self.__changeable_parameter_names.extend(self.get_changeable_parameters())

        # --- log
        self.__check_parameter = True
        self.__is_setup = True
        if enable_log:
            logger.info(f"--- {env.config.name}, {self.get_name()}")
            logger.info(f"max_episode_steps      : {env.max_episode_steps}")
            logger.info(f"player_num             : {env.player_num}")
            logger.info(f"request_env_render_mode: {self.request_env_render}")
            logger.info(f"action_space (RL requires '{self.__rl_act_type}')")
            logger.info(f" original: {env.action_space}")
            logger.info(f" env     : {self.__env_act_space}")
            logger.info(f" rl      : {self.__rl_act_space}")
            logger.info(f"observation_space (RL requires '{self.__rl_obs_type}')")
            logger.info(f" original    : {env.observation_space}")
            logger.info(f" env         : {self.__env_obs_space_in_rl}")
            if self.window_length > 1:
                logger.info(f" rl(one_step): {self.__rl_obs_space_one_step}")
            logger.info(f" rl          : {self.__rl_obs_space}")
            if self.use_render_image_state():
                if self.render_image_window_length > 1:
                    logger.info(f" render_img(one_step): {self.__rl_obs_render_img_space_one_step}")
                logger.info(f" render_img  : {self.__rl_obs_render_img_space}")

    def _get_rl_space(self, required_type: RLBaseTypes, override_type: RLBaseTypes, env_space: SpaceBase, division_num: int):
        # 優先度
        # 1. override
        # 2. RL base type
        # 3. 複数ある場合はその中からenv側の変換が最も少ないのを選択
        if override_type != RLBaseTypes.NONE:
            if not (required_type & override_type):
                logger.warning(f"An attempt is being made to overwrite a type that is not defined. override: {override_type}, required: {RLBaseTypes.to_list(required_type)}")
            required_type = override_type

        if bin(required_type.value).count("1") > 1:
            for stype in env_space.get_encode_list():
                if required_type & stype:
                    required_type = stype
                    break
            else:
                logger.warning(f"Undefined space. {env_space}")
                required_type = RLBaseTypes.NONE

        # --- rl DISCRETE
        if required_type in [
            RLBaseTypes.DISCRETE,
            RLBaseTypes.ARRAY_DISCRETE,
        ]:
            # RLがDISCRETEを要求する場合、(ENVがCONTINUOUSなら)分割する
            env_space.create_division_tbl(division_num)

        rl_space = env_space.create_encode_space(required_type, self)
        return rl_space, required_type

    # --- setup property
    def get_applied_processors(self) -> List["RLProcessor"]:
        return self.__applied_processors

    def get_applied_render_image_processors(self) -> List["RLProcessor"]:
        return self.__applied_render_img_processors

    @property
    def request_env_render(self) -> RenderModeType:
        return self.__request_env_render

    @property
    def observation_type(self) -> RLBaseTypes:
        return self.__rl_obs_type

    @property
    def observation_space_one_step(self) -> TObsSpace:
        # window length==1の時のspaceを返す
        return cast(TObsSpace, self.__rl_obs_space_one_step)

    @property
    def observation_space(self) -> TObsSpace:
        return cast(TObsSpace, self.__rl_obs_space)

    @property
    def observation_space_of_env(self) -> SpaceBase:
        return self.__env_obs_space_in_rl

    @property
    def obs_render_img_space_one_step(self) -> BoxSpace:
        return self.__rl_obs_render_img_space_one_step

    @property
    def obs_render_img_space(self) -> BoxSpace:
        return self.__rl_obs_render_img_space

    @property
    def action_type(self) -> RLBaseTypes:
        return self.__rl_act_type

    @property
    def action_space(self) -> TActSpace:
        return cast(TActSpace, self.__rl_act_space)

    @property
    def action_space_of_env(self) -> SpaceBase:
        return self.__env_act_space

    def __setattr__(self, name: str, value):
        if name in ["__is_setup"]:
            object.__setattr__(self, name, value)
            return

        # --- パラメータが決まった後の書き換え
        if getattr(self, "_check_parameter", False) and (not name.startswith("_")):
            if not hasattr(self, name):
                logger.warning(f"An undefined variable was assigned. {name}={value}")
            elif getattr(self, "_is_setup", False):
                if name not in getattr(self, "_changeable_parameter_names", []):
                    s = "A non-changeable parameter was rewritten. (This is after setup, so there may be inconsistencies in the settings.)"
                    s += f" '{name}' : '{getattr(self, name)}' -> '{value}'"
                    logger.info(s)

        object.__setattr__(self, name, value)

    # ----------------------------
    # encode/decode
    # ----------------------------
    def state_encode_one_step(self, env_state: EnvObservationType, env: EnvRun) -> RLObservationType:
        assert self.__is_setup

        # --- observation_mode
        if self.observation_mode == "":
            pass
        elif self.observation_mode == "render_image":
            if self.__request_env_render == "rgb_array":
                env_state = cast(EnvObservationType, env.render_rgb_array())
            elif self.__request_env_render == "terminal":
                env_state = cast(EnvObservationType, env.render_terminal_text_to_image())
            else:
                raise NotSupportedError(self.__request_env_render)
        else:
            raise UndefinedError(self.observation_mode)

        if self.enable_state_encode:
            for p in self.__applied_remap_obs_processors:
                env_state = p[0].remap_observation(env_state, p[1], p[2], env_run=env, rl_config=self)

            rl_state: RLObservationType = self.__env_obs_space_in_rl.encode_to_space(
                env_state,
                self.__rl_obs_space_one_step,
            )
        else:
            rl_state = cast(RLObservationType, env_state)

        if self.enable_assertion:
            assert self.__rl_obs_space_one_step.check_val(rl_state)
        elif self.enable_sanitize:
            rl_state = self.__rl_obs_space_one_step.sanitize(rl_state)

        return rl_state

    def render_image_state_encode_one_step(self, env: EnvRun) -> np.ndarray:
        assert self.__is_setup

        if self.__request_env_render == "rgb_array":
            img_state = env.render_rgb_array()
        elif self.__request_env_render == "terminal":
            img_state = env.render_terminal_text_to_image()
        else:
            raise NotSupportedError(self.__request_env_render)

        if self.enable_state_encode:
            for p in self.__applied_remap_render_img_processors:
                img_state = p[0].remap_observation(img_state, p[1], p[2], env_run=env, rl_config=self)

        if self.enable_assertion:
            assert self.__rl_obs_render_img_space_one_step.check_val(img_state)
        elif self.enable_sanitize:
            img_state = self.__rl_obs_render_img_space_one_step.sanitize(img_state)

        return cast(np.ndarray, img_state)

    def action_encode(self, env_action: EnvActionType) -> RLActionType:
        assert self.__is_setup

        if self.enable_action_decode:
            rl_act = self.__env_act_space.encode_to_space(
                env_action,
                self.__rl_act_space,
            )
        else:
            rl_act = cast(RLActionType, env_action)

        if self.enable_assertion:
            assert self.__rl_act_space.check_val(rl_act)
        elif self.enable_sanitize:
            self.__action = self.__rl_act_space.sanitize(rl_act)

        return rl_act

    def action_decode(self, rl_action: RLActionType) -> EnvActionType:
        assert self.__is_setup

        if self.enable_assertion:
            assert self.__rl_act_space.check_val(rl_action), f"{rl_action=}"
        elif self.enable_sanitize:
            self._action = self.__rl_act_space.sanitize(rl_action)

        if self.enable_action_decode:
            env_act = self.__env_act_space.decode_from_space(rl_action, self.__rl_act_space)
        else:
            env_act = cast(EnvActionType, rl_action)

        return env_act

    # ----------------------------
    # make
    # ----------------------------
    def make_memory(self, env: Optional[EnvRun] = None):
        """make_memory(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_memory

        return make_memory(self, env=env)

    def make_parameter(self, env: Optional[EnvRun] = None):
        """make_parameter(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_parameter

        return make_parameter(self, env=env)

    def make_trainer(self, parameter: "RLParameter", memory: "RLMemory", env: Optional[EnvRun] = None):
        """make_trainer(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_trainer

        return make_trainer(self, parameter=parameter, memory=memory, env=env)

    def make_worker(
        self,
        env: EnvRun,
        parameter: Optional["RLParameter"] = None,
        memory: Optional["RLMemory"] = None,
    ):
        """make_worker(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_worker

        return make_worker(self, env=env, parameter=parameter, memory=memory)

    def make_workers(
        self,
        players: PlayersType,
        env: EnvRun,
        parameter: Optional["RLParameter"] = None,
        memory: Optional["RLMemory"] = None,
        main_worker: Optional["WorkerRun"] = None,
    ):
        """make_workers() と同じ動作"""
        from srl.base.rl.registration import make_workers

        return make_workers(players, env, self, parameter, memory, main_worker)

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def name(self) -> str:
        return self.get_name()

    @property
    def used_device_tf(self) -> str:
        return self.__used_device_tf

    @property
    def used_device_torch(self) -> str:
        return self.__used_device_torch

    def to_dict(self, skip_private: bool = True) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if skip_private and k.startswith("_"):
                continue
            d[k] = v
        dat: dict = convert_for_json(d)
        return dat

    def copy(self, reset_env_config: bool = False) -> Any:
        config = self.__class__()
        config.__check_parameter = False

        for k, v in self.__dict__.items():
            if k == "__check_parameter":
                continue
            if isinstance(v, EnvRun):
                continue
            try:
                setattr(config, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        if reset_env_config:
            config.__is_setup = False
        else:
            config.__is_setup = self.__is_setup
        return config

    def get_metadata(self) -> dict:
        d = {
            "name": self.get_name(),
            "base_action_type": self.get_base_action_type(),
            "base_observation_type": self.get_base_observation_type(),
            "framework": self.get_framework(),
            "changeable_parameters": self.get_changeable_parameters(),
            "use_backup_restore": self.use_backup_restore(),
            "use_render_image_state": self.use_render_image_state(),
            "setup": self.__is_setup,
        }
        if self.__is_setup:
            d["applied_processors"] = self.get_applied_processors()
            d["applied_render_image_processors"] = self.get_applied_render_image_processors()
            d["request_env_render"] = self.request_env_render
        return d

    def summary(self):
        print("--- RLConfig ---\n" + pprint.pformat(self.to_dict()))
        print("--- Algorithm settings ---\n" + pprint.pformat(self.get_metadata()))

    def model_summary(self, expand_nested: bool = True, **kwargs):
        assert self.__is_setup
        parameter = self.make_parameter()
        parameter.summary(expand_nested=expand_nested, **kwargs)
        return parameter


@dataclass
class DummyRLConfig(RLConfig):
    name: str = ""

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.NONE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.NONE

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return self.name
