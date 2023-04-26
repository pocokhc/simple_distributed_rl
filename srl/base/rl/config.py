import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

from srl.base.define import EnvObservationType, RLActionType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class RLConfig(ABC):
    processors: List[Processor] = field(default_factory=list)
    override_env_observation_type: EnvObservationType = EnvObservationType.UNKNOWN
    override_rl_action_type: RLActionType = RLActionType.ANY  # RL側がANYの場合のみ有効
    action_division_num: int = 5
    # observation_division_num: int = 10
    extend_worker: Optional[Type["ExtendWorker"]] = None
    window_length: int = 1
    dummy_state_val: float = 0.0
    parameter_path: str = ""
    remote_memory_path: str = ""
    use_rl_processor: bool = True  # RL側のprocessorを使用するか
    change_observation_render_image: bool = False  # 状態の入力をrender_imageに変更

    # render option
    font_name: str = ""
    font_size: int = 12

    def __post_init__(self) -> None:
        self._is_set_env_config = False

        # The device used by the framework.
        self.used_device_tf: str = "/CPU"
        self.used_device_torch: str = "cpu"

    def assert_params(self) -> None:
        assert self.window_length > 0

    def get_use_framework(self) -> str:
        return ""

    # ----------------------------
    # RL config
    # ----------------------------
    @abstractmethod
    def getName(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_type(self) -> RLActionType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> RLObservationType:
        raise NotImplementedError()

    def set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        pass  # NotImplemented

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        pass  # NotImplemented

    def set_processor(self) -> List[Processor]:
        return []  # NotImplemented

    # ----------------------------
    # reset config
    # ----------------------------
    def reset(self, env: EnvRun) -> None:
        if self._is_set_env_config:
            return

        # env property
        self.env_max_episode_steps = env.max_episode_steps
        self.env_player_num = env.player_num

        self._env_action_space = env.action_space
        env_observation_space = env.observation_space
        env_observation_type = env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKNOWN:
            env_observation_type = self.override_env_observation_type

        # processor
        self._run_processors = []
        if self.change_observation_render_image:
            from srl.base.rl.processors.render_image_processor import RenderImageProcessor

            self._run_processors.append(RenderImageProcessor())
        self._run_processors.extend(self.processors)
        if self.use_rl_processor:
            self._run_processors.extend(self.set_processor())
        for processor in self._run_processors:
            env_observation_space, env_observation_type = processor.change_observation_info(
                env_observation_space,
                env_observation_type,
                self.observation_type,
                env,
            )

        # window_length
        self._one_observation_shape = env_observation_space.observation_shape
        if self.window_length > 1:
            env_observation_space = BoxSpace((self.window_length,) + self._one_observation_shape)

        self._env_observation_space = env_observation_space
        self._env_observation_type = env_observation_type

        # action division
        if isinstance(self._env_action_space, BoxSpace) and self.action_type == RLActionType.DISCRETE:
            self._env_action_space.set_action_division(self.action_division_num)

        # observation division
        # 状態は分割せずに四捨五入
        # if (
        #    isinstance(self._env_observation_space, BoxSpace)
        #    and self.observation_type == RLActionType.DISCRETE
        #    and self.env_observation_type == EnvObservationType.CONTINUOUS
        # ):
        #    self._env_observation_space.set_division(self.observation_division_num)

        self.set_config_by_env(env, self._env_action_space, env_observation_space, env_observation_type)
        self._is_set_env_config = True

        logger.info(f"max_episode_steps     : {self.env_max_episode_steps}")
        logger.info(f"player_num            : {self.env_player_num}")
        logger.info(f"action_space(env)     : {env.action_space}")
        logger.info(f"action_space(rl)      : {self._env_action_space}")
        logger.info(f"observation_type(env) : {env.observation_type}")
        logger.info(f"observation_type(rl)  : {self.env_observation_type}")
        logger.info(f"observation_space(env): {env.observation_space}")
        logger.info(f"observation_space(rl) : {self._env_observation_space}")

    def __setattr__(self, name, value):
        if name == "_is_set_env_config":
            object.__setattr__(self, name, value)
            return

        # configが書き変わったら reset が必要
        if name in [
            "processors",
            "override_env_observation_type",
            "action_division_num",
            "window_length",
            "change_observation_render_image",
            "use_rl_processor",
        ]:
            self._is_set_env_config = False
        object.__setattr__(self, name, value)

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def is_set_env_config(self) -> bool:
        return self._is_set_env_config

    @property
    def action_space(self) -> SpaceBase:
        return self._env_action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._env_observation_space

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._env_observation_space.observation_shape

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type

    def copy(self, reset_env_config: bool = False) -> "RLConfig":
        config = self.__class__()

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

    def set_parameter(self, update_params: dict):
        self.__dict__.update(update_params)
