import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Type, Union, cast

import numpy as np

import srl
from srl.base.define import (
    ObservationModes,
    PlayerType,
    RenderModes,
    RLBaseActTypes,
    RLBaseObsTypes,
    SpaceTypes,
    TActSpace,
    TObsSpace,
)
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.exception import NotSupportedError, UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker
    from srl.base.rl.memory import IRLMemoryWorker, RLMemory
    from srl.base.rl.parameter import RLParameter
    from srl.base.rl.processor import RLProcessor
    from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC, Generic[TActSpace, TObsSpace]):
    """RLConfig はアルゴリズムの動作を定義します。
    アルゴリズム毎に別々のハイパーパラメータがありますが、ここはアルゴリズム共通のパラメータの定義となります。
    """

    #: 状態の入力を指定
    observation_mode: Union[str, ObservationModes] = ObservationModes.ENV

    #: env の observation_type を上書きします。
    #: 例えばgymの自動判定で想定外のTypeになった場合、ここで上書きできます。
    override_observation_type: SpaceTypes = SpaceTypes.UNKNOWN
    #: action_type を上書きします。
    override_action_type: Union[str, RLBaseActTypes] = RLBaseActTypes.NONE

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    action_division_num: int = 10

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    observation_division_num: int = 1000

    #: 1stepあたり、環境内で余分に進めるstep数
    #: 例えばframeskip=3の場合、1step実行すると、環境内では4frame進みます。
    frameskip: int = 0

    #: ExtendWorkerを使う場合に指定
    extend_worker: Optional[Type["ExtendWorker"]] = None
    #: 指定されていた場合、Parameter生成時にpathファイルをロードします
    parameter_path: str = ""
    #: 指定されていた場合、Memory生成時にpathファイルをロードします
    memory_path: str = ""
    #: Trueの場合、アルゴリズム側で指定されたprocessorを使用します
    use_rl_processor: bool = True

    #: Processorを使う場合、定義したProcessorのリスト
    processors: List["RLProcessor"] = field(default_factory=list)
    #: Processorを使う場合、定義したProcessorのリスト
    render_image_processors: List["RLProcessor"] = field(default_factory=list)

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

    # --- other
    #: action/observationの値をエラーが出ないように可能な限り変換します。
    #: ※エラー終了の可能性は減りますが、値の変換等による予期しない動作を引き起こす可能性が高くなります
    enable_sanitize: bool = True
    #: action/observationの値を厳密にチェックし、おかしい場合は例外を出力します。
    #: enable_assertionが有効な場合は、enable_sanitizeは無効です。
    enable_assertion: bool = False
    #: dtype
    dtype = np.float32

    def __post_init__(self) -> None:
        self._is_setup = False
        self._is_setup_env = ""

        # The device used by the framework.
        self._used_device_tf: str = "/CPU"
        self._used_device_torch: str = "cpu"

        self._used_rgb_array: bool = False

        self._changeable_parameter_names_base = [
            "parameter_path",
            "memory_path",
            "enable_sanitize",
            "enable_assertion",
        ]
        self._changeable_parameter_names: List[str] = []
        self._check_parameter = True  # last

    def assert_params(self) -> None:
        assert self.window_length > 0

    def create_scheduler(self) -> "SchedulerConfig":
        from srl.rl.schedulers.scheduler import SchedulerConfig

        return SchedulerConfig()

    # ----------------------------
    # RL config
    # ----------------------------
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_base_action_type(self) -> RLBaseActTypes:
        # discrete or continuousは選択式, boxのimageはenv
        raise NotImplementedError()

    @abstractmethod
    def get_base_observation_type(self) -> RLBaseObsTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_framework(self) -> str:
        return ""

    def setup_from_env(self, env: EnvRun) -> None:
        pass  # NotImplemented

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def get_processors(self) -> List["RLProcessor"]:
        return []  # NotImplemented

    def get_render_image_processors(self) -> List["RLProcessor"]:
        return []  # NotImplemented

    def get_changeable_parameters(self) -> List[str]:
        return []  # NotImplemented

    def use_backup_restore(self) -> bool:
        return False

    def use_render_image_state(self) -> bool:
        return False

    # ----------------------------
    # setup
    # ----------------------------
    def is_setup(self, env_name: str) -> bool:
        if self._is_setup_env != env_name:
            return False
        return True

    def setup(self, env: EnvRun, enable_log: bool = True) -> None:
        if self._is_setup_env != env.name:
            self._is_setup = False
            self._is_setup_env = ""
        if self._is_setup:
            return
        self._check_parameter = False
        self.observation_mode = ObservationModes.from_str(self.observation_mode)
        self.override_action_type = RLBaseActTypes.from_str(self.override_action_type)

        # env property
        self.env_max_episode_steps = env.max_episode_steps
        self.env_player_num = env.player_num

        # --- backup/restore check
        if self.get_used_backup_restore():
            try:
                env.setup()
                env.reset()
                d = env.backup()
                env.restore(d)
            except Exception:
                logger.error(f"'{self.get_name()}' uses restore/backup, but it is not implemented in {env.name}.")
                raise

        # -------------------------------------------------
        # processor
        # -------------------------------------------------
        self._obs_processors: List[RLProcessor] = []
        self._render_img_processors: List[RLProcessor] = []
        self._episode_processors: List[RLProcessor] = []

        # user processor
        for p in self.processors:
            if hasattr(p, "remap_observation"):
                self._obs_processors.append(p.copy())
            self._episode_processors.append(p.copy())
        for p in self.render_image_processors:
            if hasattr(p, "remap_observation"):
                self._render_img_processors.append(p.copy())

        # rl processor
        if self.use_rl_processor:
            for p in self.get_processors():
                if p is None:
                    continue
                if hasattr(p, "remap_observation"):
                    self._obs_processors.append(p.copy())
                self._episode_processors.append(p.copy())
            for p in self.get_render_image_processors():
                if p is None:
                    continue
                if hasattr(p, "remap_observation"):
                    self._render_img_processors.append(p.copy())

        [p.setup(env, self) for p in self._episode_processors]

        # -------------------------------------------------
        # observation space, 4種類
        #  1. env space(original)
        #  2. overrideやprocessor後のspace(env_obs_spaces_in_rl)
        #  3. 2をRLへ変換した後の rl obs space
        #  4. 3を(window length>1) windows length後のspace()
        # -------------------------------------------------

        # --- observation_mode による変更
        if self.observation_mode == ObservationModes.ENV:
            env_obs_space = env.observation_space.copy()

            # --- observation_typeの上書き
            if isinstance(env_obs_space, BoxSpace):
                if self.override_observation_type != SpaceTypes.UNKNOWN:
                    if enable_log:
                        s = "override observation type: "
                        s += f"{env.observation_space} -> {self.override_observation_type}"
                        logger.info(s)
                    env_obs_space._stype = self.override_observation_type
        elif self.observation_mode == ObservationModes.RENDER_IMAGE:
            env.setup(**srl.RunContext(render_mode=RenderModes.rgb_array).to_dict())
            env.reset()
            rgb_array = env.render_rgb_array()
            if rgb_array is not None:
                self.obs_render_type = "rgb_array"
            else:
                rgb_array = env.render_terminal_text_to_image()
                if rgb_array is not None:
                    self.obs_render_type = "terminal"
                else:
                    raise NotSupportedError("Failed to get image.")
            env_obs_space = BoxSpace(rgb_array.shape, 0, 255, np.uint8, SpaceTypes.COLOR)
            self._used_rgb_array = True
        else:
            raise UndefinedError(self.observation_mode)

        # --- apply processors
        if self.enable_state_encode:
            for p in self._obs_processors:
                p.setup(env, self)
                new_env_obs_space = p.remap_observation_space(env_obs_space, env, self)
                if enable_log and (env_obs_space != new_env_obs_space):
                    logger.info(f"apply obs processor: {repr(p)}")
                    logger.info(f"   {env_obs_space}")
                    logger.info(f" ->{new_env_obs_space}")
                env_obs_space = new_env_obs_space

        # one step はここで確定
        self._env_obs_space_in_rl = env_obs_space

        # --- obs type & space
        # 優先度
        # 1. RL base type
        # 2. env obs space type
        base_type = self.get_base_observation_type()

        # --- rl one step space
        if base_type == RLBaseObsTypes.DISCRETE:
            # RLがDISCRETEで(ENVがCONTINUOUSなら)分割してDISCRETEにする
            self._env_obs_space_in_rl.create_division_tbl(self.observation_division_num)
            create_space = "ArrayDiscreteSpace"
        elif base_type == RLBaseObsTypes.BOX:
            create_space = "BoxSpace_float"
        else:
            create_space = ""

        # create space
        self._rl_obs_space_one_step = self._env_obs_space_in_rl.create_encode_space(create_space)

        # --- window_length
        if self.window_length > 1:
            self._rl_obs_space = self._rl_obs_space_one_step.create_stack_space(self.window_length)
        else:
            self._rl_obs_space = self._rl_obs_space_one_step

        # --- include render image
        if self.use_render_image_state():
            env.setup(**srl.RunContext(render_mode=RenderModes.rgb_array).to_dict())
            env.reset()
            rgb_array = env.render_rgb_array()
            if rgb_array is not None:
                self.obs_render_type = "rgb_array"
            else:
                rgb_array = env.render_terminal_text_to_image()
                if rgb_array is not None:
                    self.obs_render_type = "terminal"
                else:
                    raise NotSupportedError("Failed to get image.")
            self._rl_obs_render_img_space_one_step: BoxSpace = BoxSpace(
                rgb_array.shape, 0, 255, np.uint8, SpaceTypes.COLOR
            )
            self._used_rgb_array = True

            if self.enable_state_encode:
                for p in self._render_img_processors:
                    p.setup(env, self)
                    new_space = p.remap_observation_space(self._rl_obs_render_img_space_one_step, env, self)
                    if enable_log and (self._rl_obs_render_img_space_one_step != new_space):
                        logger.info(f"apply img obs processor: {repr(p)}")
                        logger.info(f"   {new_space}")
                        logger.info(f" ->{self._rl_obs_render_img_space_one_step}")
                    self._rl_obs_render_img_space_one_step = cast(BoxSpace, new_space)

            if self.window_length > 1:
                self._rl_obs_render_img_space = self._rl_obs_render_img_space_one_step.create_stack_space(
                    self.window_length
                )
            else:
                self._rl_obs_render_img_space = self._rl_obs_render_img_space_one_step

        # -----------------------
        # action space, 2種類、特に前処理とかはないのでそのままenvと同じになる
        # 1. env space(original)
        # 2. rl action space
        # -----------------------
        self._env_act_space = env.action_space.copy()

        # --- act type & space
        # 優先度
        # 1. override_action_type
        # 2. RL base type
        # 3. env action_space type
        self._rl_act_type: RLBaseActTypes = self.get_base_action_type()
        if self.override_action_type != RLBaseActTypes.NONE:
            assert isinstance(self.override_action_type, RLBaseActTypes)
            self._rl_act_type = self.override_action_type

        # 複数flagがある場合はenvに近いもの
        if bin(self._rl_act_type.value).count("1") > 1:
            priority_list = []
            if self._env_act_space.is_discrete():
                priority_list = [
                    RLBaseActTypes.DISCRETE,
                    RLBaseActTypes.CONTINUOUS,
                    RLBaseActTypes.NONE,
                ]
            elif self._env_act_space.is_continuous():
                priority_list = [
                    RLBaseActTypes.CONTINUOUS,
                    RLBaseActTypes.DISCRETE,
                    RLBaseActTypes.NONE,
                ]
            elif self._env_act_space.is_image():
                if self._env_act_space.stype == SpaceTypes.GRAY_2ch:
                    priority_list = [
                        RLBaseActTypes.DISCRETE,
                        RLBaseActTypes.CONTINUOUS,
                        RLBaseActTypes.NONE,
                    ]
                elif self._env_act_space.stype == SpaceTypes.GRAY_3ch:
                    priority_list = [
                        RLBaseActTypes.DISCRETE,
                        RLBaseActTypes.CONTINUOUS,
                        RLBaseActTypes.NONE,
                    ]
                elif self._env_act_space.stype == SpaceTypes.COLOR:
                    priority_list = [
                        RLBaseActTypes.DISCRETE,
                        RLBaseActTypes.CONTINUOUS,
                        RLBaseActTypes.NONE,
                    ]
                elif self._env_act_space.stype == SpaceTypes.IMAGE:
                    priority_list = [
                        RLBaseActTypes.DISCRETE,
                        RLBaseActTypes.CONTINUOUS,
                        RLBaseActTypes.NONE,
                    ]
                else:
                    raise UndefinedError(self._env_act_space.stype)
            else:
                raise UndefinedError(self._env_act_space.stype)

            for stype in priority_list:
                if self._rl_act_type & stype:
                    self._rl_act_type = stype
                    break
            else:
                logger.warning(f"Undefined space. {self._env_act_space}")
                self._rl_act_type = RLBaseActTypes.NONE

        # --- space
        if self._rl_act_type == RLBaseActTypes.DISCRETE:
            # RLがDISCRETEで(ENVがCONTINUOUSなら)分割してDISCRETEにする
            self._env_act_space.create_division_tbl(self.action_division_num)
            create_space = "DiscreteSpace"
        elif self._rl_act_type == RLBaseActTypes.CONTINUOUS:
            create_space = "ArrayContinuousSpace"
        else:
            create_space = ""

        # create space
        self._rl_act_space = self._env_act_space.create_encode_space(create_space)

        # --------------------------------------------

        # --- option
        self.setup_from_env(env)

        # --- changeable parameters
        self._changeable_parameter_names = self._changeable_parameter_names_base[:]
        self._changeable_parameter_names.extend(self.get_changeable_parameters())

        # --- log
        self._check_parameter = True
        self._is_setup = True
        self._is_setup_env = env.name
        if enable_log:
            logger.info(f"--- {env.config.name}, {self.get_name()}")
            logger.info(f"max_episode_steps      : {env.max_episode_steps}")
            logger.info(f"player_num             : {env.player_num}")
            logger.info(f"action_space (RL requires '{self.get_base_action_type()}' type)")
            logger.info(f" original: {env.action_space}")
            logger.info(f" env     : {self._env_act_space}")
            logger.info(f" rl      : {self._rl_act_space}")
            logger.info(f" rl_type : {self._rl_act_type}")
            logger.info(f"observation_space (RL requires '{self.get_base_observation_type()}' type)")
            logger.info(f" original    : {env.observation_space}")
            logger.info(f" env         : {self._env_obs_space_in_rl}")
            if self.window_length > 1:
                logger.info(f" rl(one_step): {self._rl_obs_space_one_step}")
            logger.info(f" rl          : {self._rl_obs_space}")
            if self.use_render_image_state():
                if self.window_length > 1:
                    logger.info(f" render_img(one_step): {self._rl_obs_render_img_space_one_step}")
                logger.info(f" render_img          : {self._rl_obs_render_img_space}")

    # --- setup property
    def get_run_observation_processors(self) -> List["RLProcessor"]:
        return self._obs_processors

    def get_run_render_image_processors(self) -> List["RLProcessor"]:
        return self._render_img_processors

    def get_run_episode_processors(self) -> List["RLProcessor"]:
        return self._episode_processors

    @property
    def observation_space_one_step(self) -> TObsSpace:
        # window length==1の時のspaceを返す
        return cast(TObsSpace, self._rl_obs_space_one_step)

    @property
    def observation_space(self) -> TObsSpace:
        return cast(TObsSpace, self._rl_obs_space)

    @property
    def observation_space_of_env(self) -> SpaceBase:
        return self._env_obs_space_in_rl

    @property
    def obs_render_img_space_one_step(self) -> BoxSpace:
        return self._rl_obs_render_img_space_one_step

    @property
    def obs_render_img_space(self) -> BoxSpace:
        return self._rl_obs_render_img_space

    @property
    def action_type(self) -> RLBaseActTypes:
        return self._rl_act_type

    @property
    def action_space(self) -> TActSpace:
        return cast(TActSpace, self._rl_act_space)

    @property
    def action_space_of_env(self) -> SpaceBase:
        return self._env_act_space

    @property
    def used_rgb_array(self) -> bool:
        return self._used_rgb_array

    def __setattr__(self, name: str, value):
        if name in ["_is_setup", "_is_setup_env"]:
            object.__setattr__(self, name, value)
            return

        # --- パラメータが決まった後の書き換え
        if getattr(self, "_check_parameter", False) and (not name.startswith("_")):
            if not hasattr(self, name):
                logger.warning(f"An undefined variable was assigned. {name}={value}")
            elif getattr(self, "_is_setup", False):
                if name not in getattr(self, "_changeable_parameter_names", []):
                    s = f"Parameter has been rewritten. '{name}' : '{getattr(self, name)}' -> '{value}'"
                    logger.info(s)

        object.__setattr__(self, name, value)

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def name(self) -> str:
        return self.get_name()

    @property
    def used_device_tf(self) -> str:
        return self._used_device_tf

    @property
    def used_device_torch(self) -> str:
        return self._used_device_torch

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
            config._is_setup_env = ""
        else:
            config._is_setup = self._is_setup
            config._is_setup_env = self._is_setup_env
        return config

    def make_memory(self, env: Optional[EnvRun] = None, is_load: bool = True):
        """make_memory(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_memory

        return make_memory(self, env=env, is_load=is_load)

    def make_parameter(self, env: Optional[EnvRun] = None, is_load: bool = True):
        """make_parameter(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_parameter

        return make_parameter(self, env=env, is_load=is_load)

    def make_trainer(self, parameter: "RLParameter", memory: "RLMemory", env: Optional[EnvRun] = None):
        """make_trainer(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_trainer

        return make_trainer(self, parameter=parameter, memory=memory, env=env)

    def make_worker(
        self,
        env: EnvRun,
        parameter: Optional["RLParameter"] = None,
        memory: Optional["IRLMemoryWorker"] = None,
    ):
        """make_worker(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_worker

        return make_worker(self, env=env, parameter=parameter, memory=memory)

    def make_workers(
        self,
        players: List[PlayerType],
        env: EnvRun,
        parameter: Optional["RLParameter"] = None,
        memory: Optional["IRLMemoryWorker"] = None,
    ):
        """make_workers() と同じ動作"""
        from srl.base.rl.registration import make_workers

        return make_workers(players, env, self, parameter, memory)


@dataclass
class DummyRLConfig(RLConfig):
    name: str = ""

    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.NONE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.NONE

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return self.name
