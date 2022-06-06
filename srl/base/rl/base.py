import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from srl.base.define import (
    EnvAction,
    EnvObservation,
    EnvObservationType,
    Info,
    RLAction,
    RLActionType,
    RLInvalidAction,
    RLObservation,
    RLObservationType,
)
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor, RuleBaseProcessor

logger = logging.getLogger(__name__)


class RLConfig(ABC):
    def __init__(self) -> None:
        self.processors: List[Processor] = []
        self.override_env_observation_type: EnvObservationType = EnvObservationType.UNKNOWN
        self.action_division_num: int = 5
        # self.observation_division_num: int = 10
        self.rulebase: Optional[RuleBaseProcessor] = None

    @staticmethod
    @abstractmethod
    def getName() -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_type(self) -> RLActionType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> RLObservationType:
        raise NotImplementedError()

    @abstractmethod
    def _set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        raise NotImplementedError()

    def set_config_by_env(self, env: EnvRun) -> None:
        self._env_action_space = env.action_space
        env_observation_space = env.observation_space
        env_observation_type = env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKNOWN:
            env_observation_type = self.override_env_observation_type

        # processor
        for processor in self.processors:
            env_observation_space, env_observation_type = processor.change_observation_info(
                env_observation_space,
                env_observation_type,
                self.observation_type,
                env.get_original_env(),
            )
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

        self._set_config_by_env(env, self._env_action_space, env_observation_space, env_observation_type)
        self._is_set_config_by_env = True

    @property
    def is_set_config_by_env(self) -> bool:
        return hasattr(self, "_is_set_config_by_env")

    @property
    def env_action_space(self) -> SpaceBase:
        return self._env_action_space

    @property
    def env_observation_space(self) -> SpaceBase:
        return self._env_observation_space

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type

    def assert_params(self) -> None:
        pass  # do nothing

    def copy(self) -> "RLConfig":
        # TODO
        return pickle.loads(pickle.dumps(self))


class RLParameter(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        logger.debug(f"save: {path}")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str) -> None:
        logger.debug(f"load: {path}")
        with open(path, "rb") as f:
            self.restore(pickle.load(f))

    def summary(self):
        pass  # do nothing


class RLRemoteMemory(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        logger.debug(f"save: {path}")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
        except Exception:
            if os.path.isfile(path):
                os.remove(path)

    def load(self, path: str) -> None:
        logger.debug(f"load: {path}")
        with open(path, "rb") as f:
            self.restore(pickle.load(f))


class RLTrainer(ABC):
    def __init__(self, config: RLConfig, parameter: RLParameter, remote_memory: RLRemoteMemory):
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory

    @abstractmethod
    def get_train_count(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        raise NotImplementedError()


class RLWorker(ABC):
    def __init__(
        self,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        worker_id: int = 0,
    ):
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.worker_id = worker_id
        self._training = False
        self._distributed = False

    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed

    # ------------------------------
    # util functions
    # ------------------------------
    def observation_encode(self, state: EnvObservation, env: EnvRun) -> RLObservation:
        for processor in self.config.processors:
            state = processor.process_observation(state, env.get_original_env())

        if self.config.observation_type == RLObservationType.DISCRETE:
            state = self.config.env_observation_space.observation_discrete_encode(state)
        elif self.config.observation_type == RLObservationType.CONTINUOUS:
            state = self.config.env_observation_space.observation_continuous_encode(state)
        else:
            state = np.asarray(state)
        return state

    def action_encode(self, action: EnvAction) -> RLInvalidAction:
        # discrete only
        if self.config.action_type == RLActionType.DISCRETE:
            action = self.config.env_action_space.action_discrete_encode(action)
        return action  # type: ignore

    def action_decode(self, action: RLAction) -> EnvAction:
        if self.config.action_type == RLActionType.DISCRETE:
            assert not isinstance(action, list)
            action = int(action)
            env_action = self.config.env_action_space.action_discrete_decode(action)
        elif self.config.action_type == RLActionType.CONTINUOUS:
            if isinstance(action, list):
                action = [float(a) for a in action]
            else:
                action = [float(action)]
            env_action = self.config.env_action_space.action_continuous_decode(action)
        else:
            env_action = action
        return env_action

    def set_training(self, training: bool, distributed: bool) -> None:
        self._training = training
        self._distributed = distributed

    def get_invalid_actions(self, env: EnvRun) -> List[RLInvalidAction]:
        return [self.action_encode(a) for a in env.get_invalid_actions(self.player_index)]

    # ------------------------------
    # implement
    # ------------------------------
    @abstractmethod
    def _call_on_reset(self, status: RLObservation, env: EnvRun) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _call_policy(self, status: RLObservation, env: EnvRun) -> RLAction:
        raise NotImplementedError()

    @abstractmethod
    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
    ) -> Info:
        raise NotImplementedError()

    @abstractmethod
    def _call_render(self, env: EnvRun) -> None:
        raise NotImplementedError()

    # ------------------------------------
    # episode functions
    # ------------------------------------
    def on_reset(self, env: EnvRun, player_index: int) -> None:
        self.player_index = player_index
        self.info = None
        self._is_reset = False
        self._step_reward = 0
        self._is_rl_policy = False

    def policy(self, env: EnvRun) -> EnvAction:
        # state
        state = self.observation_encode(env.state, env)

        # ルールベースの割り込み
        action = None
        if self.config.rulebase is not None:
            action = self.config.rulebase.process_policy_before(
                env,
                self.player_index,
            )

        self._is_rl_policy = False
        if action is None:

            # 初期化していないなら初期化する
            if not self._is_reset:
                self._call_on_reset(state, env)
                self._is_reset = True

            # rl policy
            action = self._call_policy(state, env)
            action = self.action_decode(action)
            self._is_rl_policy = True

            # ルールベース割り込み
            if self.config.rulebase is not None:
                action = self.config.rulebase.process_policy_after(
                    action,
                    env,
                    self.player_index,
                )

        return action

    def on_step(self, env: EnvRun) -> Optional[Info]:

        # 初期化前はskip
        if not self._is_reset:
            return None

        # 相手の番のrewardも加算
        self._step_reward += env.step_rewards[self.player_index]

        # 次の自分の番の前の状態がon_step、だたしdoneは実行する
        if not env.done and self.player_index not in env.next_player_indices:
            return None

        # 自分の番、rlの場合on_stepを実行
        if self._is_rl_policy:
            next_state = self.observation_encode(env.state, env)
            self._info = self._call_on_step(next_state, self._step_reward, env.done, env)
            self._step_reward = 0

        return self._info

    def render(self, env: EnvRun) -> None:
        if self._is_rl_policy and self._is_reset:
            self._call_render(env)
        elif self.config.rulebase is not None:
            self.config.rulebase.policy_render(env, self.player_index)

    # ------------------------------------
    # utils
    # ------------------------------------
    def env_step(self, env: EnvRun) -> Tuple[np.ndarray, float]:
        # 次の自分の番になるまで進める(相手のpolicyは自分)

        reward = 0
        while True:
            state = self.observation_encode(env.state, env)
            actions = [
                self.action_decode(self._env_step_policy(state, env)) if i in env.next_player_indices else None
                for i in range(env.player_num)
            ]
            env.step(actions)
            reward += env.step_rewards[self.player_index]

            if self.player_index in env.next_player_indices:
                break
            if env.done:
                break

        return self.observation_encode(env.state, env), reward

    def _env_step_policy(self, state: np.ndarray, env: EnvRun) -> RLAction:
        raise NotImplementedError()
