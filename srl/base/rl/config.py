import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Type, Union, cast

import numpy as np

from srl.base.define import ObservationModes, PlayerType, RenderModes, RLBaseTypes, SpaceTypes, TActSpace, TObsSpace
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.text import TextSpace
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker
    from srl.base.rl.memory import IRLMemoryTrainer, IRLMemoryWorker
    from srl.base.rl.parameter import RLParameter
    from srl.base.rl.processor import Processor
    from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC, Generic[TActSpace, TObsSpace]):
    """RLConfig はアルゴリズムの動作を定義します。
    アルゴリズム毎に別々のハイパーパラメータがありますが、ここはアルゴリズム共通のパラメータの定義となります。
    """

    #: 状態の入力を指定、複数指定するとMultipleSpaceになる
    observation_mode: Union[str, ObservationModes] = ObservationModes.ENV  # type: ignore , type OK

    #: env の observation_type を上書きします。
    #: 例えばgymの自動判定で想定外のTypeになった場合、ここで上書きできます。
    override_observation_type: SpaceTypes = SpaceTypes.UNKNOWN
    #: action_type を上書きします。
    override_action_type: SpaceTypes = SpaceTypes.UNKNOWN

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    #: The number of divisions when converting from continuous to discrete values.
    #: If -1, round by round transform.
    action_division_num: int = 5

    #: 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。
    #: The number of divisions when converting from continuous to discrete values.
    #: If -1, round by round transform.
    observation_division_num: int = -1

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
    processors: List["Processor"] = field(default_factory=list)

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

        # The device used by the framework.
        self._used_device_tf: str = "/CPU"
        self._used_device_torch: str = "cpu"

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
    def get_base_action_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_base_observation_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_framework(self) -> str:
        return ""

    def setup_from_env(self, env: EnvRun) -> None:
        pass  # NotImplemented

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def get_processors(self) -> List[Optional["Processor"]]:
        return []  # NotImplemented

    def get_changeable_parameters(self) -> List[str]:
        return []  # NotImplemented

    def get_used_backup_restore(self) -> bool:
        return False

    # ----------------------------
    # setup
    # ----------------------------
    def setup(self, env: EnvRun, enable_log: bool = True) -> None:
        if self._is_setup:
            return
        self._check_parameter = False
        self.observation_mode: ObservationModes = ObservationModes.from_str(self.observation_mode)

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
        obs_processors: List[Processor] = []
        self._episode_processors: List[Processor] = []

        # user processor
        for p in self.processors:
            if hasattr(p, "remap_observation"):
                obs_processors.append(p)
            self._episode_processors.append(p.copy())

        # rl processor
        if self.use_rl_processor:
            for p in self.get_processors():
                if p is None:
                    continue
                if hasattr(p, "remap_observation"):
                    obs_processors.append(p)
                self._episode_processors.append(p.copy())

        [p.setup(env, self) for p in self._episode_processors]

        # -------------------------------------------------
        # observation space, 4種類
        #  1. env space(original)
        #  2. overrideやprocessor後のspace(env_obs_spaces_in_rl)
        #  3. 2をRLへ変換した後の rl obs space
        #  4. 3を(window length>1) windows length後のspace()
        # -------------------------------------------------
        env_obs_space_list: List[SpaceBase] = []
        self._is_env_obs_multi = False
        _is_env_obs_multi2 = False

        # 2つ以上ならマルチ
        if self.observation_mode not in [
            ObservationModes.ENV,
            ObservationModes.RENDER_IMAGE,
            ObservationModes.RENDER_TERMINAL,
        ]:
            _is_env_obs_multi2 = True

        # --- ObsMode
        if self.observation_mode & ObservationModes.ENV:
            env_obs_space = env.observation_space.copy()
            if isinstance(env_obs_space, MultiSpace):
                self._is_env_obs_multi = True
                _is_env_obs_multi2 = True
                if self.override_observation_type != SpaceTypes.UNKNOWN and enable_log:
                    logger.info("override_observation_type is not supported in MultiSpace.")
                env_obs_space_list.extend(env_obs_space.spaces)
            else:
                # --- observation_typeの上書き
                if isinstance(env_obs_space, BoxSpace):
                    if self.override_observation_type != SpaceTypes.UNKNOWN:
                        if enable_log:
                            s = "override observation type: "
                            s += f"{env.observation_space} -> {self.override_observation_type}"
                            logger.info(s)
                        env_obs_space._stype = self.override_observation_type
                env_obs_space_list.append(env_obs_space)

        if self.observation_mode & ObservationModes.RENDER_IMAGE:
            from srl.base.context import RunContext

            env.setup(RunContext(render_mode=RenderModes.rgb_array))
            env.reset()
            rgb_array = env.render_rgb_array()
            env_obs_space_list.append(BoxSpace(rgb_array.shape, 0, 255, np.uint8, SpaceTypes.COLOR))

        if self.observation_mode & ObservationModes.RENDER_TERMINAL:
            from srl.base.context import RunContext

            env.setup(RunContext(render_mode=RenderModes.ansi))
            env.reset()
            texts = env.render_ansi()
            env_obs_space_list.append(TextSpace(len(texts)))

        # --- apply processors
        self._obs_processors_list: List[List[Processor]] = []
        if self.enable_state_encode:
            # 各spaceに適用する
            for i in range(len(env_obs_space_list)):
                p_list = []
                for p in obs_processors:
                    p = p.copy()
                    p.setup(env, self)
                    new_env_obs_space = p.remap_observation_space(env_obs_space_list[i], env, self)
                    if enable_log and (env_obs_space_list[i] != new_env_obs_space):
                        logger.info(f"apply obs processor: {repr(p)}")
                        logger.info(f"   {env_obs_space_list[i]}")
                        logger.info(f" ->{new_env_obs_space}")
                    env_obs_space_list[i] = new_env_obs_space
                    p_list.append(p)
                self._obs_processors_list.append(p_list)

        # one step はここで確定
        assert len(env_obs_space_list) > 0
        if _is_env_obs_multi2:
            self._env_obs_space_in_rl = MultiSpace(env_obs_space_list)
        else:
            self._env_obs_space_in_rl = env_obs_space_list[0]

        # --- obs type & space
        # 優先度
        # 1. RL base type
        # 2. env obs space type
        base_type = self.get_base_observation_type()

        # --- type
        if base_type == RLBaseTypes.DISCRETE:
            rl_type = SpaceTypes.DISCRETE
        elif base_type == RLBaseTypes.CONTINUOUS:
            rl_type = SpaceTypes.CONTINUOUS
        elif base_type == RLBaseTypes.IMAGE:
            if SpaceTypes.is_image(self._env_obs_space_in_rl.stype):
                rl_type = self._env_obs_space_in_rl.stype
            else:
                raise NotSupportedError("This algorithm only supports image formats.")
        elif base_type == RLBaseTypes.MULTI:
            rl_type = SpaceTypes.MULTI
        elif base_type == RLBaseTypes.NONE:
            rl_type = self._env_obs_space_in_rl.stype
        else:
            # 1 IMAGE
            # - TEXT
            # 2 CONTINUOUS
            # 3 DISCRETE
            # 4 MULTI
            if (base_type & RLBaseTypes.IMAGE) and SpaceTypes.is_image(self._env_obs_space_in_rl.stype):
                rl_type = self._env_obs_space_in_rl.stype
            elif self._env_obs_space_in_rl.stype == SpaceTypes.CONTINUOUS:
                if base_type & RLBaseTypes.CONTINUOUS:
                    rl_type = SpaceTypes.CONTINUOUS
                elif base_type & RLBaseTypes.DISCRETE:
                    rl_type = SpaceTypes.DISCRETE
                elif base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    logger.warning(f"Undefined space. {self._env_obs_space_in_rl}")
                    rl_type = SpaceTypes.UNKNOWN
            elif self._env_obs_space_in_rl.stype == SpaceTypes.DISCRETE:
                if base_type & RLBaseTypes.DISCRETE:
                    rl_type = SpaceTypes.DISCRETE
                elif base_type & RLBaseTypes.CONTINUOUS:
                    rl_type = SpaceTypes.CONTINUOUS
                elif base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    logger.warning(f"Undefined space. {self._env_obs_space_in_rl}")
                    rl_type = SpaceTypes.UNKNOWN
            elif isinstance(self._env_obs_space_in_rl, MultiSpace):
                if base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    if self._env_obs_space_in_rl.is_discrete:
                        if base_type & RLBaseTypes.DISCRETE:
                            rl_type = SpaceTypes.DISCRETE
                        elif base_type & RLBaseTypes.CONTINUOUS:
                            rl_type = SpaceTypes.CONTINUOUS
                        else:
                            logger.warning(f"Undefined space. {self._env_act_space}")
                            rl_type = SpaceTypes.UNKNOWN
                    else:
                        if base_type & RLBaseTypes.CONTINUOUS:
                            rl_type = SpaceTypes.CONTINUOUS
                        elif base_type & RLBaseTypes.DISCRETE:
                            rl_type = SpaceTypes.DISCRETE
                        else:
                            logger.warning(f"Undefined space. {self._env_act_space}")
                            rl_type = SpaceTypes.UNKNOWN
            else:
                logger.warning(f"Undefined space. {self._env_obs_space_in_rl}")
                rl_type = SpaceTypes.UNKNOWN

        # --- space
        def _create_rl_obs_space(rl_type: SpaceTypes, env_space: SpaceBase) -> SpaceBase:
            if rl_type == SpaceTypes.UNKNOWN:
                return env_space
            if rl_type == SpaceTypes.DISCRETE:
                # --- division
                # RLがDISCRETEで(ENVがCONTINUOUSなら)分割してDISCRETEにする
                env_space.create_division_tbl(self.observation_division_num)
                return ArrayDiscreteSpace(
                    env_space.list_int_size,
                    env_space.list_int_low,
                    env_space.list_int_high,
                )
            elif rl_type == SpaceTypes.CONTINUOUS:
                return BoxSpace(
                    env_space.np_shape,
                    env_space.np_low,
                    env_space.np_high,
                    self.dtype,
                    rl_type,
                )
            elif SpaceTypes.is_image(rl_type):
                return BoxSpace(
                    env_space.np_shape,
                    env_space.np_low,
                    env_space.np_high,
                    self.dtype,
                    rl_type,
                )
            elif rl_type == SpaceTypes.MULTI:
                if isinstance(env_space, MultiSpace):
                    _spaces = []
                    for space in env_space.spaces:
                        _spaces.append(_create_rl_obs_space(space.stype, space))
                    return MultiSpace(_spaces)
                else:
                    return MultiSpace([_create_rl_obs_space(env_space.stype, env_space)])
            else:
                logger.warning(f"Undefined space. {env_space}")
                return env_space

        self._rl_obs_space_one_step = _create_rl_obs_space(rl_type, self._env_obs_space_in_rl)

        # --- window_length
        if self.window_length > 1:
            self._rl_obs_space = self._rl_obs_space_one_step.create_stack_space(self.window_length)
        else:
            self._rl_obs_space = self._rl_obs_space_one_step

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
        base_type = self.get_base_action_type()

        # --- type
        if self.override_action_type != SpaceTypes.UNKNOWN:
            rl_type = self.override_action_type
        elif base_type == RLBaseTypes.CONTINUOUS:
            rl_type = SpaceTypes.CONTINUOUS
        elif base_type == RLBaseTypes.IMAGE:
            if SpaceTypes.is_image(self._env_act_space.stype):
                rl_type = self._env_act_space.stype
            else:
                raise NotSupportedError("This algorithm only supports image formats.")
        elif base_type == RLBaseTypes.MULTI:
            rl_type = SpaceTypes.MULTI
        elif base_type == RLBaseTypes.NONE:
            rl_type = self._env_act_space.stype
        else:
            # 1 IMAGE
            # - TEXT
            # 2 DISCRETE
            # 3 CONTINUOUS
            # 4 MULTI
            if (base_type & RLBaseTypes.IMAGE) and SpaceTypes.is_image(self._env_act_space.stype):
                rl_type = self._env_act_space.stype
            elif self._env_act_space.stype == SpaceTypes.DISCRETE:
                if base_type & RLBaseTypes.DISCRETE:
                    rl_type = SpaceTypes.DISCRETE
                elif base_type & RLBaseTypes.CONTINUOUS:
                    rl_type = SpaceTypes.CONTINUOUS
                elif base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    logger.warning(f"Undefined space. {self._env_act_space}")
                    rl_type = SpaceTypes.UNKNOWN
            elif self._env_act_space.stype == SpaceTypes.CONTINUOUS:
                if base_type & RLBaseTypes.CONTINUOUS:
                    rl_type = SpaceTypes.CONTINUOUS
                elif base_type & RLBaseTypes.DISCRETE:
                    rl_type = SpaceTypes.DISCRETE
                elif base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    logger.warning(f"Undefined space. {self._env_act_space}")
                    rl_type = SpaceTypes.UNKNOWN
            elif isinstance(self._env_act_space, MultiSpace):
                if base_type & RLBaseTypes.MULTI:
                    rl_type = SpaceTypes.MULTI
                else:
                    if self._env_act_space.is_discrete:
                        if base_type & RLBaseTypes.DISCRETE:
                            rl_type = SpaceTypes.DISCRETE
                        elif base_type & RLBaseTypes.CONTINUOUS:
                            rl_type = SpaceTypes.CONTINUOUS
                        else:
                            logger.warning(f"Undefined space. {self._env_act_space}")
                            rl_type = SpaceTypes.UNKNOWN
                    else:
                        if base_type & RLBaseTypes.CONTINUOUS:
                            rl_type = SpaceTypes.CONTINUOUS
                        elif base_type & RLBaseTypes.DISCRETE:
                            rl_type = SpaceTypes.DISCRETE
                        else:
                            logger.warning(f"Undefined space. {self._env_act_space}")
                            rl_type = SpaceTypes.UNKNOWN
            else:
                logger.warning(f"Undefined space. {self._env_act_space}")
                rl_type = SpaceTypes.UNKNOWN

        # --- space
        def _create_rl_act_space(rl_type: SpaceTypes, env_space: SpaceBase) -> SpaceBase:
            if rl_type == SpaceTypes.UNKNOWN:
                return env_space
            if rl_type == SpaceTypes.DISCRETE:
                # --- division
                # RLがDISCRETEで(ENVがCONTINUOUSなら)分割してDISCRETEにする
                env_space.create_division_tbl(self.action_division_num)
                return DiscreteSpace(env_space.int_size)
            elif rl_type == SpaceTypes.CONTINUOUS:
                return ArrayContinuousSpace(
                    env_space.list_float_size,
                    env_space.list_float_low,
                    env_space.list_float_high,
                    self.dtype,
                )
            elif SpaceTypes.is_image(rl_type):
                return BoxSpace(
                    env_space.np_shape,
                    env_space.np_low,
                    env_space.np_high,
                    self.dtype,
                    stype=rl_type,
                )
            elif rl_type == SpaceTypes.MULTI:
                if isinstance(env_space, MultiSpace):
                    _spaces = []
                    for space in env_space.spaces:
                        _spaces.append(_create_rl_act_space(space.stype, space))
                    return MultiSpace(_spaces)
                else:
                    return MultiSpace([_create_rl_act_space(env_space.stype, env_space)])
            else:
                logger.warning(f"Undefined space. {env_space}")
                return env_space

        self._rl_act_space = _create_rl_act_space(rl_type, self._env_act_space)

        # --------------------------------------------

        # --- option
        self.setup_from_env(env)

        # --- changeable parameters
        self._changeable_parameter_names = self._changeable_parameter_names_base[:]
        self._changeable_parameter_names.extend(self.get_changeable_parameters())

        # --- log
        self._check_parameter = True
        self._is_setup = True
        if enable_log:
            logger.info(f"--- {env.config.name}, {self.get_name()}")
            logger.info(f"max_episode_steps      : {env.max_episode_steps}")
            logger.info(f"player_num             : {env.player_num}")
            logger.info(f"action_space (RL requires '{self.get_base_action_type()}' type)")
            logger.info(f" original: {env.action_space}")
            logger.info(f" env     : {self._env_act_space}")
            logger.info(f" rl      : {self._rl_act_space}")
            logger.info(f"observation_space (RL requires '{self.get_base_observation_type()}' type)")
            logger.info(f" original    : {env.observation_space}")
            logger.info(f" env         : {self._env_obs_space_in_rl}")
            if self.window_length > 1:
                logger.info(f" rl(one_step): {self._rl_obs_space_one_step}")
            logger.info(f" rl          : {self._rl_obs_space}")

    # --- setup property

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    @property
    def observation_processors_list(self) -> List[List["Processor"]]:
        return self._obs_processors_list

    @property
    def episode_processors(self) -> List["Processor"]:
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
    def action_space(self) -> TActSpace:
        return cast(TActSpace, self._rl_act_space)

    @property
    def action_space_of_env(self) -> SpaceBase:
        return self._env_act_space

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
        else:
            config._is_setup = self._is_setup
        return config

    def make_memory(self, env: Optional[EnvRun] = None, is_load: bool = True):
        """make_memory(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_memory

        return make_memory(self, env=env, is_load=is_load)

    def make_parameter(self, env: Optional[EnvRun] = None, is_load: bool = True):
        """make_parameter(rl_config) と同じ動作"""
        from srl.base.rl.registration import make_parameter

        return make_parameter(self, env=env, is_load=is_load)

    def make_trainer(self, parameter: "RLParameter", memory: "IRLMemoryTrainer", env: Optional[EnvRun] = None):
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

        return make_workers(players, env, rl_config=self, parameter=parameter, memory=memory)


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
