import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Type

import numpy as np

from srl.base.define import EnvObservationTypes, RLObservationType, RLTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.processor import Processor
from srl.base.spaces.box import BoxSpace
from srl.utils import common

if TYPE_CHECKING:
    from srl.base.rl.algorithms.extend_worker import ExtendWorker


logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC):
    processors: List[Processor] = field(default_factory=list)
    override_env_observation_type: EnvObservationTypes = EnvObservationTypes.UNKNOWN
    override_action_type: RLTypes = RLTypes.ANY  # RL側がANYの場合のみ有効

    action_division_num: int = 5
    """
    The number of divisions when converting from continuous to discrete values.
    If -1, round by round transform.
    """
    # 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。

    observation_division_num: int = -1
    """
    The number of divisions when converting from continuous to discrete values.
    If -1, round by round transform.
    """
    # 連続値から離散値に変換する場合の分割数です。-1の場合round変換で丸めます。

    extend_worker: Optional[Type["ExtendWorker"]] = None
    parameter_path: str = ""
    remote_memory_path: str = ""
    use_rl_processor: bool = True  # RL側のprocessorを使用するか

    use_render_image_for_observation: bool = False
    """ Change state input to render_image. Existing settings will be overwritten. """
    # 状態の入力をrender_imageに変更。既存の設定は上書きされます。

    # --- Worker Config
    enable_state_encode: bool = True
    enable_action_decode: bool = True
    enable_reward_encode: bool = True
    window_length: int = 1
    dummy_state_val: float = 0.0

    # --- other
    enable_sanitize_value: bool = True
    enable_assertion_value: bool = False

    def __post_init__(self) -> None:
        self._is_set_env_config = False
        self._run_processors: List[Processor] = []

        # The device used by the framework.
        self._used_device_tf: str = "/CPU"
        self._used_device_torch: str = "cpu"

        self._check_parameter = True

    def assert_params(self) -> None:
        assert self.window_length > 0

    def get_use_framework(self) -> Optional[str]:
        if not hasattr(self, "framework"):
            return None
        framework = getattr(self, "framework")
        if framework == "tf":
            return "tensorflow"
        if framework == "":
            if common.is_package_installed("tensorflow"):
                framework = "tensorflow"
        if framework == "":
            if common.is_package_installed("torch"):
                framework = "torch"
        assert framework != "", "'tensorflow' or 'torch' could not be found."
        return framework

    # ----------------------------
    # RL config
    # ----------------------------
    @abstractmethod
    def getName(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_action_type(self) -> RLTypes:
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_observation_type(self) -> RLTypes:
        raise NotImplementedError()

    def set_config_by_env(self, env: EnvRun) -> None:
        pass  # NotImplemented

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def set_processor(self) -> List[Processor]:
        return []  # NotImplemented

    @property
    def info_types(self) -> dict:
        """infoの情報のタイプを指定、出力形式等で使用を想定
        各行の句は省略可能
        name : {
            "type": 型を指定(None, int, float, str)
            "data": 以下のデータ形式を指定
                "ave" : 平均値を使用(default)
                "last": 最後のデータを使用
                "min" : 最小値
                "max" : 最大値
        }
        """
        return {}  # NotImplemented

    # ----------------------------
    # reset config
    # ----------------------------
    def reset(self, env: EnvRun) -> None:
        if self._is_set_env_config:
            return
        self._check_parameter = False

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

        # -----------------------
        # observation
        # -----------------------

        # --- observation_typeの上書き
        if self.override_env_observation_type != EnvObservationTypes.UNKNOWN:
            rl_env_observation_type = self.override_env_observation_type
            logger.info(f"override observation type: {rl_env_observation_type}")

        self._run_processors = []
        if self.enable_state_encode:
            # --- add processor
            if self.use_render_image_for_observation:
                from srl.rl.processors.render_image_processor import RenderImageProcessor

                self._run_processors.append(RenderImageProcessor())
            self._run_processors.extend(self.processors)
            if self.use_rl_processor:
                self._run_processors.extend(self.set_processor())

            # --- processor
            for processor in self._run_processors:
                rl_observation_space, rl_env_observation_type = processor.preprocess_observation_space(
                    rl_observation_space,
                    rl_env_observation_type,
                    env,
                    self,
                )
                logger.info(f"processor obs space: {rl_observation_space}")
                logger.info(f"processor obs type : {rl_env_observation_type}")

        # --- window_length
        self._one_observation = rl_observation_space
        if self.window_length > 1:
            rl_observation_space = BoxSpace(
                (self.window_length,) + self._one_observation.shape,
                np.min(self._one_observation.low),
                np.max(self._one_observation.high),
            )
            logger.info(f"window_length obs space: {rl_observation_space}")

        self._rl_observation_space = rl_observation_space
        self._rl_env_observation_type = rl_env_observation_type

        # --- obs type
        # 優先度
        # 1. RL
        # 2. obs_space
        rl_obs_type = self.base_observation_type
        if rl_obs_type == RLTypes.ANY:
            rl_obs_type = self._rl_observation_space.rl_type
        self._rl_observation_type = rl_obs_type

        # check type
        _f = False
        if rl_obs_type == RLTypes.DISCRETE:
            if rl_env_observation_type not in [
                EnvObservationTypes.DISCRETE,
                EnvObservationTypes.SHAPE3,
                EnvObservationTypes.SHAPE2,
            ]:
                _f = True
        if _f:
            logger.warning(f"EnvType and RLType do not match. {rl_env_observation_type} != {rl_obs_type}")

        # -----------------------
        #  action type
        # -----------------------
        # 優先度
        # 1. RL
        # 2. override_action_type
        # 3. action_space
        rl_action_type = self.base_action_type
        if rl_action_type == RLTypes.ANY:
            rl_action_type = self.override_action_type
        if rl_action_type == RLTypes.ANY:
            rl_action_type = self._env_action_space.rl_type
        self._rl_action_type = rl_action_type

        # --- base obs type
        base_obs_type = self.base_observation_type
        if base_obs_type == RLTypes.ANY:
            base_obs_type = self._rl_observation_space.rl_type

        # --- division
        # RLが DISCRETE で Space が CONTINUOUS なら分割して DISCRETE にする
        if (self._rl_action_type == RLTypes.DISCRETE) and (self._env_action_space.rl_type == RLTypes.CONTINUOUS):
            self._env_action_space.create_division_tbl(self.action_division_num)
        if (base_obs_type == RLTypes.DISCRETE) and (self._rl_observation_space.rl_type == RLTypes.CONTINUOUS):
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

        # --- option
        self.set_config_by_env(env)

        self._is_set_env_config = True
        logger.info(f"action_space(env)       : {self._env_action_space}")
        logger.info(f"action_type(rl)         : {self._rl_action_type}")
        logger.info(f"observation_env_type(rl): {self._rl_env_observation_type}")
        logger.info(f"observation_type(rl)    : {self._rl_observation_type}")
        logger.info(f"observation_space(rl)   : {self._rl_observation_space}")

    def __setattr__(self, name, value):
        if name == "_is_set_env_config":
            object.__setattr__(self, name, value)
            return

        if hasattr(self, "_check_parameter"):
            if self._check_parameter and not hasattr(self, name):
                logger.warning(f"An undefined variable was assigned. {name}={value}")

        # configが書き変わったら reset が必要
        if name in [
            "processors",
            "override_env_observation_type",
            "override_action_type",
            "action_division_num",
            "use_render_image_for_observation",
            "use_rl_processor",
            "enable_state_encode",
            "enable_action_decode",
            "window_length",
        ]:
            self._is_set_env_config = False
        object.__setattr__(self, name, value)

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def name(self) -> str:
        return self.getName()

    @property
    def is_set_env_config(self) -> bool:
        return self._is_set_env_config

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

    def copy(self, reset_env_config: bool = False) -> "RLConfig":
        config = self.__class__()
        config._check_parameter = False

        for k, v in self.__dict__.items():
            if isinstance(v, EnvRun):
                continue
            try:
                setattr(config, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        if reset_env_config:
            config._is_set_env_config = False
        else:
            config._is_set_env_config = self._is_set_env_config
        return config

    def set_parameter(self, update_params: dict) -> None:
        self._check_parameter = True
        self.__dict__.update(update_params)

    def create_dummy_state(self, is_one: bool = False) -> RLObservationType:
        if is_one:
            return np.full(self._one_observation.shape, self.dummy_state_val, dtype=np.float32)
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
class DummyConfig(RLConfig):
    name: str = "dummy"

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.ANY

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.ANY

    def getName(self) -> str:
        return self.name
