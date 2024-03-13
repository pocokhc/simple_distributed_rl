import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import numpy as np

from srl.base.define import EnvTypes, ObservationModes, RenderModes, RLBaseTypes, RLTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.exception import UndefinedError
from srl.base.rl.processor import EpisodeProcessor, ObservationProcessor, ProcessorType
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.text import TextSpace
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker


logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC):
    """RLConfig はアルゴリズムの動作を定義します。
    アルゴリズム毎に別々のハイパーパラメータがありますが、ここはアルゴリズム共通のパラメータの定義となります。
    """

    #: 状態の入力を指定、複数指定するとMultipleSpaceになる
    observation_mode: ObservationModes = ObservationModes.ENV

    #: env_observation_type を上書きできます。
    #: 例えばgymの自動判定で想定外のTypeになった場合、ここで上書きできます。
    override_env_observation_type: EnvTypes = EnvTypes.UNKNOWN
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

    #: Processorを使う場合、定義したProcessorのリスト
    processors: List[ProcessorType] = field(default_factory=list)

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

    @abstractmethod
    def get_base_action_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_base_observation_type(self) -> RLBaseTypes:
        raise NotImplementedError()

    @abstractmethod
    def get_use_framework(self) -> str:
        raise NotImplementedError()

    def set_config_by_env(self, env: EnvRun) -> None:
        pass  # NotImplemented

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return []  # NotImplemented

    def get_changeable_parameters(self) -> List[str]:
        return []  # NotImplemented

    def get_used_backup_restore(self) -> bool:
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
    def get_info_types(self) -> dict:
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
            logger.info(f"action_space(env)     : {env.action_space}")
            logger.info(f"observation_space(env): {env.observation_space}")

        # env property
        self.env_max_episode_steps = env.max_episode_steps
        self.env_player_num = env.player_num

        # --- backup/restore check
        if self.get_used_backup_restore():
            try:
                env.reset()
                d = env.backup()
                env.restore(d)
            except Exception:
                logger.error(f"'{self.getName()}' uses restore/backup, but it is not implemented in {env.name}.")
                raise

        # -------------------------------------------------
        # processor
        # -------------------------------------------------
        obs_processors: List[ObservationProcessor] = []
        self._episode_processors: List[EpisodeProcessor] = []

        # user processor
        for p in self.processors:
            if isinstance(p, ObservationProcessor):
                obs_processors.append(p)
            elif isinstance(p, EpisodeProcessor):
                self._episode_processors.append(p)

        # rl processor
        if self.use_rl_processor:
            for p in self.get_processors():
                if isinstance(p, ObservationProcessor):
                    obs_processors.append(p)
                elif isinstance(p, EpisodeProcessor):
                    self._episode_processors.append(p)

        # -------------------------------------------------
        # observation
        # -------------------------------------------------
        env_obs_spaces: List[SpaceBase] = []
        self._is_env_obs_multi = False
        if self.observation_mode & ObservationModes.ENV:
            env_obs_space = env.observation_space.copy()
            if isinstance(env_obs_space, MultiSpace):
                self._is_env_obs_multi = True
                if self.override_env_observation_type != EnvTypes.UNKNOWN and enable_log:
                    logger.info("override_env_observation_type is not supported in MultiSpace.")
                env_obs_spaces.extend(env_obs_space.spaces)
            else:
                # --- observation_typeの上書き
                if self.override_env_observation_type != EnvTypes.UNKNOWN:
                    if enable_log:
                        s = "override env observation type: "
                        s += f"{env.observation_space.env_type} -> {self.override_env_observation_type}"
                        logger.info(s)
                    env_obs_space.set_env_type(self.override_env_observation_type)
                env_obs_spaces.append(env_obs_space)

        if self.observation_mode & ObservationModes.RENDER_IMAGE:
            env.config.override_render_mode = RenderModes.rgb_array
            env.reset(render_mode=RenderModes.rgb_array)
            rgb_array = env.render_rgb_array()
            env_obs_spaces.append(BoxSpace(rgb_array.shape, 0, 255, np.uint8, EnvTypes.COLOR))

        if self.observation_mode & ObservationModes.RENDER_TERMINAL:
            env_obs_spaces.append(TextSpace(TODO))

        # --- apply processors
        self._obs_processors_list: List[List[ObservationProcessor]] = []
        if self.enable_state_encode:
            # 各spaceに適用する
            for i in range(len(env_obs_spaces)):
                p_list = []
                for p in obs_processors:
                    p = p.copy()
                    p.setup(env, self)  # processors setup
                    new_env_obs_space = p.preprocess_observation_space(env_obs_spaces[i], env, self)
                    if enable_log and (env_obs_spaces[i] != new_env_obs_space):
                        logger.info(f"apply observation processor: {repr(p)}")
                        logger.info(f"   {env_obs_spaces[i]}")
                        logger.info(f" ->{new_env_obs_space}")
                    env_obs_spaces[i] = new_env_obs_space
                    p_list.append(p)
                self._obs_processors_list.append(p_list)

        # --- obs type
        # 優先度
        # 1. RL
        # 2. obs_space
        for space in env_obs_spaces:
            if self.get_base_observation_type() == RLBaseTypes.DISCRETE:
                rl_type = RLTypes.DISCRETE
            elif self.get_base_observation_type() == RLBaseTypes.CONTINUOUS:
                rl_type = RLTypes.CONTINUOUS
            elif self.get_base_observation_type() == RLBaseTypes.ANY:
                if space.env_type == EnvTypes.DISCRETE:
                    rl_type = RLTypes.DISCRETE
                else:
                    rl_type = RLTypes.CONTINUOUS
            else:
                raise UndefinedError(self.get_base_observation_type())

            # CONTINUOUSなら画像チェックする
            if rl_type == RLTypes.CONTINUOUS:
                if space.env_type in [
                    EnvTypes.GRAY_2ch,
                    EnvTypes.GRAY_3ch,
                    EnvTypes.COLOR,
                    EnvTypes.IMAGE,
                ]:
                    rl_type = RLTypes.IMAGE

            space.set_rl_type(rl_type)

            # --- division
            # RLが DISCRETE で Space が CONTINUOUS なら分割して DISCRETE にする
            if rl_type == RLTypes.DISCRETE and space.env_type == EnvTypes.CONTINUOUS:
                space.create_division_tbl(self.observation_division_num)

        # --- window_length
        env_obs_spaces_one_step_in_rl = env_obs_spaces
        env_obs_spaces_in_rl = env_obs_spaces
        if self.window_length > 1:
            for i in range(len(env_obs_spaces_one_step_in_rl)):
                one_space = env_obs_spaces_one_step_in_rl[i]
                env_obs_spaces_in_rl[i] = BoxSpace(
                    (self.window_length,) + one_space.shape,
                    np.min(one_space.low),
                    np.max(one_space.high),
                    one_space.dtype,
                    one_space.env_type,
                )
                env_obs_spaces_in_rl[i].set_rl_type(one_space.rl_type)

        # --- multi space
        # obsはMultiが1つなら展開される
        assert len(env_obs_spaces_in_rl) > 0
        if len(env_obs_spaces_in_rl) > 1:
            self._env_obs_space_in_rl: SpaceBase = MultiSpace(env_obs_spaces_in_rl)
        else:
            self._env_obs_space_in_rl: SpaceBase = env_obs_spaces_in_rl[0]
        self._env_obs_space_one_step_in_rl: List[SpaceBase] = env_obs_spaces_one_step_in_rl

        # -----------------------
        #  action
        # -----------------------
        env_act_spaces: List[SpaceBase] = []
        env_act_space = env.action_space.copy()
        if isinstance(env_act_space, MultiSpace):
            env_act_spaces.extend(env_act_space.spaces)
        else:
            env_act_spaces.append(env_act_space)

        # --- act type
        # 優先度
        # 1. RL
        # 2. override_action_type
        # 3. action_space
        for space in env_act_spaces:
            if self.get_base_action_type() == RLBaseTypes.DISCRETE:
                rl_type = RLTypes.DISCRETE
            elif self.get_base_action_type() == RLBaseTypes.CONTINUOUS:
                rl_type = RLTypes.CONTINUOUS
            elif self.get_base_action_type() == RLBaseTypes.ANY:
                if self.override_action_type != RLTypes.UNKNOWN:
                    rl_type = self.override_action_type
                else:
                    if space.env_type == EnvTypes.DISCRETE:
                        rl_type = RLTypes.DISCRETE
                    else:
                        rl_type = RLTypes.CONTINUOUS
            else:
                raise UndefinedError(self.get_base_action_type())

            # CONTINUOUSなら画像チェックする
            if rl_type == RLTypes.CONTINUOUS:
                if space.env_type in [
                    EnvTypes.GRAY_2ch,
                    EnvTypes.GRAY_3ch,
                    EnvTypes.COLOR,
                    EnvTypes.IMAGE,
                ]:
                    rl_type = RLTypes.IMAGE

            space.set_rl_type(rl_type)

            # --- division
            # RLが DISCRETE で Space が CONTINUOUS なら分割して DISCRETE にする
            if rl_type == RLTypes.DISCRETE and space.env_type == EnvTypes.CONTINUOUS:
                space.create_division_tbl(self.action_division_num)

        # --- multi space
        # actはMultiが1つでも保持される
        assert len(env_act_spaces) > 0
        if isinstance(env.action_space, MultiSpace):
            self._env_act_space_in_rl: SpaceBase = MultiSpace(env_act_spaces)
        elif len(env_act_spaces) > 1:
            self._env_act_space_in_rl: SpaceBase = MultiSpace(env_act_spaces)
        else:
            self._env_act_space_in_rl: SpaceBase = env_act_spaces[0]

        # ------------------------------

        # --- set rl property
        if self._env_act_space_in_rl.rl_type == RLTypes.DISCRETE:
            self._action_num = self.action_space.n
            self._action_low = np.ndarray(0)
            self._action_high = np.ndarray(self._action_num - 1)
        else:
            # ANYの場合もCONTINUOUS
            self._action_num = self.action_space.list_size
            self._action_low = np.array(self.action_space.list_low)
            self._action_high = np.array(self.action_space.list_high)

        # --- option
        self.set_config_by_env(env)

        # --- changeable parameters
        self._changeable_parameter_names = self._changeable_parameter_names_base[:]
        self._changeable_parameter_names.extend(self.get_changeable_parameters())

        # --- log
        self._check_parameter = True
        self._is_setup = True
        if enable_log:
            logger.info(f"action_space(rl)     : {self._env_act_space_in_rl}")
            logger.info(f"observation_space(rl): {self._env_obs_space_in_rl}")
            if self.window_length > 1:
                logger.info(f"observation_spaces_one_step(rl): {self._env_obs_space_one_step_in_rl}")

    # --- setup property

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    @property
    def observation_processors_list(self) -> List[List[ObservationProcessor]]:
        return self._obs_processors_list

    @property
    def episode_processors(self) -> List[EpisodeProcessor]:
        return self._episode_processors

    @property
    def is_env_obs_multi(self) -> bool:
        return self._is_env_obs_multi

    @property
    def action_space(self) -> SpaceBase:
        return self._env_act_space_in_rl

    @property
    def action_type(self) -> RLTypes:
        return self._env_act_space_in_rl.rl_type

    @property
    def observation_spaces_one_step(self) -> List[SpaceBase]:
        return self._env_obs_space_one_step_in_rl

    @property
    def observation_space(self) -> SpaceBase:
        return self._env_obs_space_in_rl

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._env_obs_space_in_rl.shape

    @property
    def observation_type(self) -> RLTypes:
        return self._env_obs_space_in_rl.rl_type

    @property
    def observation_type_of_env(self) -> EnvTypes:
        return self._env_obs_space_in_rl.env_type

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
    def used_device_tf(self) -> str:
        return self._used_device_tf

    @property
    def used_device_torch(self) -> str:
        return self._used_device_torch

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

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    def get_use_framework(self) -> str:
        return ""

    def getName(self) -> str:
        return self.name
