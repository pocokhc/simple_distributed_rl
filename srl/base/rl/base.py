import binascii
import io
import logging
import lzma
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from srl.base.define import (
    EnvAction,
    EnvObservation,
    EnvObservationType,
    Info,
    RLAction,
    RLActionType,
    RLObservation,
    RLObservationType,
)
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


class RLConfig(ABC):
    def __init__(
        self,
        processors: List[Processor] = None,
        override_env_observation_type: EnvObservationType = EnvObservationType.UNKNOWN,
        action_division_num: int = 5,
        extend_worker: Optional[Type["ExtendWorker"]] = None,
        window_length: int = 1,
        dummy_state_val: float = 0.0,
        parameter_path: str = "",
        remote_memory_path: str = "",
        use_rl_processor: bool = True,
    ) -> None:
        if processors is None:
            self.processors = []
        else:
            self.processors = processors
        self.override_env_observation_type = override_env_observation_type
        self.action_division_num = action_division_num
        # self.observation_division_num: int = 10
        self.extend_worker = extend_worker
        self.window_length = window_length
        self.dummy_state_val = dummy_state_val
        self.parameter_path = parameter_path
        self.remote_memory_path = remote_memory_path
        self.use_rl_processor = use_rl_processor

    def assert_params(self) -> None:
        assert self.window_length > 0

    # ----------------------------
    # RL config
    # ----------------------------
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
    def set_env(self, env: EnvRun) -> None:
        self.__env = env
        self._is_set_env = True

    def reset_config(self) -> None:
        # env property
        self.env_max_episode_steps = self.__env.max_episode_steps
        self.env_player_num = self.__env.player_num

        self._env_action_space = self.__env.action_space
        env_observation_space = self.__env.observation_space
        env_observation_type = self.__env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKNOWN:
            env_observation_type = self.override_env_observation_type

        # processor
        self._run_processors = self.processors[:]
        if self.use_rl_processor:
            self._run_processors.extend(self.set_processor())
        for processor in self._run_processors:
            env_observation_space, env_observation_type = processor.change_observation_info(
                env_observation_space,
                env_observation_type,
                self.observation_type,
                self.__env,
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

        self.set_config_by_env(self.__env, self._env_action_space, env_observation_space, env_observation_type)

        logger.debug(f"max_episode_steps     : {self.env_max_episode_steps}")
        logger.debug(f"player_num            : {self.env_player_num}")
        logger.debug(f"action_space(env)     : {self.__env.action_space}")
        logger.debug(f"action_space(rl)      : {self._env_action_space}")
        logger.debug(f"observation_type(env) : {self.__env.observation_type}")
        logger.debug(f"observation_type(rl)  : {self.env_observation_type}")
        logger.debug(f"observation_space(env): {self.__env.observation_space}")
        logger.debug(f"observation_space(rl) : {self._env_observation_space}")

    # ----------------------------
    # utils
    # ----------------------------
    @property
    def is_set_env(self) -> bool:
        return hasattr(self, "_is_set_env")

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

    def copy(self) -> "RLConfig":
        return pickle.loads(pickle.dumps(self))


class RLParameter(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any, **kwargs) -> None:
        self.call_restore(data, **kwargs)

    def backup(self, **kwargs) -> Any:
        return self.call_backup(**kwargs)

    def save(self, path: str) -> None:
        logger.debug(f"parameter save: {path}")
        try:
            t0 = time.time()
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
            logger.info(f"parameter saved({time.time() - t0:.1f}s): {path}")
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str) -> None:
        logger.info(f"parameter load: {path}")
        t0 = time.time()
        with open(path, "rb") as f:
            self.restore(pickle.load(f))
        logger.debug(f"parameter loaded({time.time() - t0:.1f}s)")

    def summary(self, **kwargs):
        pass  # NotImplemented


class RLRemoteMemory(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, dat: Any, **kwargs) -> None:
        if isinstance(dat, tuple):
            dat = lzma.decompress(dat[0])
            dat = pickle.loads(dat)
        self.call_restore(dat, **kwargs)

    def backup(self, compress: bool = False, **kwargs) -> Any:
        dat = self.call_backup(**kwargs)
        if compress:
            dat = pickle.dumps(dat)
            dat = lzma.compress(dat)
            dat = (dat, True)
        return dat

    def save(self, path: str, compress: bool = True, **kwargs) -> None:
        logger.debug(f"memory save (size: {self.length()}): {path}")
        try:
            t0 = time.time()
            dat = self.call_backup(**kwargs)
            if compress:
                dat = pickle.dumps(dat)
                with lzma.open(path, "w") as f:
                    f.write(dat)
            else:
                with open(path, "wb") as f:
                    pickle.dump(dat, f)
            logger.info(f"memory saved (size: {self.length()}, time: {time.time() - t0:.1f}s): {path}")
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str, **kwargs) -> None:
        logger.debug(f"memory load: {path}")
        t0 = time.time()
        # LZMA
        with open(path, "rb") as f:
            compress = binascii.hexlify(f.read(6)) == b"fd377a585a00"
        if compress:
            with lzma.open(path) as f:
                dat = f.read()
            dat = pickle.loads(dat)
        else:
            with open(path, "rb") as f:
                dat = pickle.load(f)
        self.call_restore(dat, **kwargs)
        logger.info(f"memory loaded (size: {self.length()}, time: {time.time() - t0:.1f}s): {path}")


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


class WorkerBase(ABC):
    # ------------------------------
    # implement
    # ------------------------------
    @property
    @abstractmethod
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        raise NotImplementedError()

    @abstractmethod
    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    # ------------------------------
    # implement(option)
    # ------------------------------
    def render_terminal(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> None:
        pass

    # ------------------------------------
    # episode
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed

    def set_play_info(self, training: bool, distributed: bool) -> None:
        self._training = training
        self._distributed = distributed


class RLWorker(WorkerBase):
    """Define Worker for RL."""

    def __init__(
        self,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ):
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.actor_id = actor_id
        self.__dummy_state = np.full(self.config._one_observation_shape, self.config.dummy_state_val)

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(self, state: EnvObservation, env: EnvRun) -> RLObservation:
        for processor in self.config._run_processors:
            state = processor.process_observation(state, env)

        if self.config.observation_type == RLObservationType.DISCRETE:
            state = self.config.observation_space.observation_discrete_encode(state)
        elif self.config.observation_type == RLObservationType.CONTINUOUS:
            state = self.config.observation_space.observation_continuous_encode(state)
        else:
            state = np.asarray(state)
        if state.shape == ():
            state = state.reshape((1,))
        return state

    def action_encode(self, action: EnvAction) -> RLAction:
        # discrete only
        if self.config.action_type == RLActionType.DISCRETE:
            action = self.config.action_space.action_discrete_encode(action)
        return action  # type: ignore

    def action_decode(self, action: RLAction) -> EnvAction:
        if self.config.action_type == RLActionType.DISCRETE:
            assert not isinstance(action, list)
            action = int(action)
            env_action = self.config.action_space.action_discrete_decode(action)
        elif self.config.action_type == RLActionType.CONTINUOUS:
            if isinstance(action, list):
                action = [float(a) for a in action]
            else:
                action = [float(action)]
            env_action = self.config.action_space.action_continuous_decode(action)
        else:
            env_action = action
        return env_action

    def reward_encode(self, reward: float, env: EnvRun) -> float:
        for processor in self.config._run_processors:
            reward = processor.process_reward(reward, env)
        return reward

    # ------------------------------
    # implement
    # ------------------------------
    @abstractmethod
    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: "WorkerRun") -> None:
        raise NotImplementedError()

    @abstractmethod
    def _call_policy(self, state: RLObservation, env: EnvRun, worker: "WorkerRun") -> RLAction:
        raise NotImplementedError()

    @abstractmethod
    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: "WorkerRun",
    ) -> Info:
        raise NotImplementedError()

    # ------------------------------------
    # episode
    # ------------------------------------
    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        self.__recent_states = [self.__dummy_state for _ in range(self.config.window_length)]
        self._player_index = worker.player_index
        self.__env = env

        state = self.state_encode(env.state, env)
        self.__recent_states.pop(0)
        self.__recent_states.append(state)

        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)

        self._call_on_reset(state, env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)
        else:
            state = self.__recent_states[-1]

        action = self._call_policy(state, env, worker)
        action = self.action_decode(action)
        return action

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        next_state = self.state_encode(env.state, env)
        reward = self.reward_encode(worker.reward, env)

        self.__recent_states.pop(0)
        self.__recent_states.append(next_state)

        # stacked state
        if self.config.window_length > 1:
            next_state = np.asarray(self.__recent_states)

        info = self._call_on_step(
            next_state,
            reward,
            env.done,
            env,
            worker,
        )
        return info

    # ------------------------------------
    # utils
    # ------------------------------------
    def get_invalid_actions(self, env=None) -> List[RLAction]:
        if env is None:
            env = self.__env
        return [self.action_encode(a) for a in self.__env.get_invalid_actions(self.player_index)]

    def get_valid_actions(self, env=None) -> List[RLAction]:
        if env is None:
            env = self.__env
        return [self.action_encode(a) for a in self.__env.get_valid_actions(self.player_index)]

    def sample_action(self, env=None) -> RLAction:
        if env is None:
            env = self.__env
        return self.action_encode(self.__env.sample(self.player_index))

    def env_step(self, env: EnvRun, action: RLAction, **kwargs) -> Tuple[np.ndarray, List[float], bool]:
        """Advance env one step

        Args:
            env (EnvRun): env
            action (RLAction): action

        Returns:
            Tuple[np.ndarray, List[float], bool]: 次の状態, 報酬, 終了したかどうか
        """
        _action = self.action_decode(action)
        env.step(_action, **kwargs)
        next_state = self.state_encode(env.state, env)
        # reward = env.step_rewards[player_index]  TODO:報酬の扱いが決まらないため保留
        rewards = env.step_rewards.tolist()
        self.__recent_states.append(next_state)
        return next_state, rewards, env.done

    # TODO: 現状使っていない、不要なら削除予定
    # def env_step_from_worker(self, env: EnvRun) -> Tuple[np.ndarray, float, bool]:
    #     """次の自分の番になるまでenvを進める
    #     (相手のpolicyは call_env_step_from_worker_policy で定義)

    #     Args:
    #         env (EnvRun): env

    #     Returns:
    #         Tuple[np.ndarray, float, bool]: 次の状態, 報酬, 終了したかどうか
    #     """

    #     reward = 0
    #     while True:
    #         state = self.observation_encode(env.state, env)
    #         action = self.call_env_step_from_worker_policy(state, env, env.next_player_index)
    #         action = self.action_decode(action)
    #         env.step(action)
    #         reward += env.step_rewards[self.player_index]

    #         if self.player_index == env.next_player_index:
    #             break
    #         if env.done:
    #             break

    #     return self.observation_encode(env.state, env), reward, env.done

    # def call_env_step_from_worker_policy(self, state: np.ndarray, env: EnvRun, player_idx: int) -> RLAction:
    #    raise NotImplementedError()


class RuleBaseWorker(WorkerBase):
    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        self._player_index = worker.player_index
        self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return self.call_on_step(env, worker)


class ExtendWorker(WorkerBase):
    def __init__(self, rl_worker: "WorkerRun", env: EnvRun):
        self.rl_worker = rl_worker
        self.env = env

    def set_play_info(self, training: bool, distributed: bool) -> None:
        super().set_play_info(training, distributed)
        self.rl_worker.set_play_info(training, distributed)

    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> None:
        self._player_index = worker.player_index
        self.rl_worker.on_reset(env, worker.player_index)
        self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> EnvAction:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        self.rl_worker.on_step(env)
        return self.call_on_step(env, worker)


class WorkerRun:
    def __init__(self, worker: WorkerBase):
        self.worker = worker

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self.worker.training

    @property
    def distributed(self) -> bool:
        return self.worker.distributed

    def set_play_info(self, training: bool, distributed: bool) -> None:
        self.worker.set_play_info(training, distributed)

    @property
    def player_index(self) -> int:
        return self._player_index

    @property
    def info(self) -> Optional[Info]:
        return self._info

    @property
    def reward(self) -> float:
        return self._step_reward

    def on_reset(self, env: EnvRun, player_index: int) -> None:
        self._player_index = player_index
        self._info = None
        self._is_reset = False
        self._step_reward = 0

    def policy(self, env: EnvRun) -> Optional[EnvAction]:
        if self.player_index != env.next_player_index:
            return None

        # 初期化していないなら初期化する
        if not self._is_reset:
            self.worker.on_reset(env, self)
            self._is_reset = True
        else:
            # 2週目以降はpolicyの実行前にstepを実行
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0

        # worker policy
        env_action = self.worker.policy(env, self)
        return env_action

    def on_step(self, env: EnvRun):
        # 初期化前はskip
        if not self._is_reset:
            return

        # 相手の番のrewardも加算
        self._step_reward += env.step_rewards[self.player_index]

        # 終了ならon_step実行
        if env.done:
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0

    def render(self, env: EnvRun, **kwargs):
        self.render_terminal(env, False, **kwargs)

    def render_terminal(self, env: EnvRun, return_text: bool = False, **kwargs):
        # 初期化前はskip
        if not self._is_reset:
            if return_text:
                return ""
            return

        if return_text:
            # 表示せずに文字列として返す
            text = ""
            _stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                self.worker.render_terminal(env, self, **kwargs)
                text = sys.stdout.getvalue()
            except NotImplementedError:
                pass
            finally:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                sys.stdout = _stdout
            return text
        else:
            try:
                self.worker.render_terminal(env, self, **kwargs)
            except NotImplementedError:
                pass
