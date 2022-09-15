import logging
import random
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from srl.base.define import (
    EnvAction,
    EnvObservation,
    Info,
    PlayRenderMode,
    RenderMode,
    RLAction,
    RLActionType,
    RLObservation,
    RLObservationType,
)
from srl.base.env.base import EnvRun
from srl.base.render import IRender, Render
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


class WorkerBase(ABC, IRender):
    def __init__(self, training: bool = False, distributed: bool = False):
        self._training = training
        self._distributed = distributed
        self.config = None

    # ------------------------------
    # implement
    # ------------------------------
    @property
    @abstractmethod
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        raise NotImplementedError()

    @abstractmethod
    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    # ------------------------------
    # implement(option)
    # ------------------------------
    def set_render_mode(self, mode: RenderMode) -> None:
        pass

    def render_terminal(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # ------------------------------------
    # episode
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed


class RLWorker(WorkerBase):
    """Define Worker for RL."""

    def __init__(
        self,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        training: bool = False,
        distributed: bool = False,
        actor_id: int = 0,
    ):
        super().__init__(training, distributed)
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.actor_id = actor_id
        self.__dummy_state = np.full(self.config._one_observation_shape, self.config.dummy_state_val)
        self.__env = None

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
    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    @abstractmethod
    def _call_policy(self, state: RLObservation, env: EnvRun, worker: "WorkerRun") -> Tuple[RLAction, Info]:
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

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        self.__recent_states = [self.__dummy_state for _ in range(self.config.window_length)]
        self._player_index = worker.player_index
        self.__env = env

        state = self.state_encode(env.state, env)
        self.__recent_states.pop(0)
        self.__recent_states.append(state)

        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)

        return self._call_on_reset(state, env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)
        else:
            state = self.__recent_states[-1]

        action, info = self._call_policy(state, env, worker)
        action = self.action_decode(action)
        return action, info

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        next_state = self.state_encode(env.state, env)
        reward = self.reward_encode(worker.reward, env)

        self.__recent_states.pop(0)
        self.__recent_states.append(next_state)

        # stacked state
        if self.config.window_length > 1:
            next_state = np.asarray(self.__recent_states)

        return self._call_on_step(
            next_state,
            reward,
            env.done,
            env,
            worker,
        )

    # ------------------------------------
    # episode info
    # ------------------------------------
    @property
    def max_episode_steps(self) -> int:
        return 0 if self.__env is None else self.__env.max_episode_steps

    @property
    def player_num(self) -> int:
        return 0 if self.__env is None else self.__env.player_num

    @property
    def step(self) -> int:
        return 0 if self.__env is None else self.__env.step_num

    # ------------------------------------
    # utils
    # ------------------------------------
    def get_invalid_actions(self, env=None) -> List[int]:
        if self.config.action_type == RLActionType.DISCRETE:
            if env is None:
                env = self.__env
            return [self.action_encode(a) for a in env.get_invalid_actions(self.player_index)]
        else:
            return []

    def get_valid_actions(self, env=None) -> List[RLAction]:
        if self.config.action_type == RLActionType.DISCRETE:
            if env is None:
                env = self.__env
            invalid_actions = self.get_invalid_actions(env)
            return [a for a in range(env.action_space.get_action_discrete_info()) if a not in invalid_actions]
        else:
            return []

    def sample_action(self, env=None) -> RLAction:
        if self.config.action_type == RLActionType.DISCRETE:
            action = random.choice(self.get_valid_actions(env))
        else:
            if env is None:
                env = self.__env
            action = env.sample(self.player_index)
            action = self.action_encode(action)
        return action

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
    def __init__(self):
        super().__init__(training=False, distributed=False)

    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        self._player_index = worker.player_index
        _t = self.call_on_reset(env, worker)
        if _t is None:
            warnings.warn("The return value of call_on_reset has changed from None to info.", DeprecationWarning)
            return {}
        return _t

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        action = self.call_policy(env, worker)
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
            action, info = action
        else:
            warnings.warn(
                "The return value of call_policy has changed from action to (action, info).", DeprecationWarning
            )
            info = {}
        return action, info

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return self.call_on_step(env, worker)


class ExtendWorker(WorkerBase):
    def __init__(self, rl_worker: "WorkerRun"):
        super().__init__(rl_worker.training, rl_worker.distributed)
        self.rl_worker = rl_worker

    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> Info:
        self._player_index = worker.player_index
        self.rl_worker.on_reset(env, worker.player_index)
        _t = self.call_on_reset(env, worker)
        if _t is None:
            warnings.warn("The return value of call_on_reset has changed from None to info.", DeprecationWarning)
            return {}
        return _t

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvAction, Info]:
        action = self.call_policy(env, worker)
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
            action, info = action
        else:
            warnings.warn(
                "The return value of call_policy has changed from action to (action, info).", DeprecationWarning
            )
            info = {}
        return action, info

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> Info:
        self.rl_worker.on_step(env)
        return self.call_on_step(env, worker)


class WorkerRun:
    def __init__(self, worker: WorkerBase):
        self.worker = worker
        self._render = Render(worker, worker.config)

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self.worker.training

    @property
    def distributed(self) -> bool:
        return self.worker.distributed

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
        logger.debug(f"worker.on_reset({player_index})")

        self._player_index = player_index
        self._info = None
        self._is_reset = False
        self._step_reward = 0

    def policy(self, env: EnvRun) -> Optional[EnvAction]:
        if self.player_index != env.next_player_index:
            return None
        logger.debug("worker.policy()")

        # 初期化していないなら初期化する
        if not self._is_reset:
            self._info = self.worker.on_reset(env, self)
            self._is_reset = True
        else:
            # 2週目以降はpolicyの実行前にstepを実行
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0

        # worker policy
        env_action, info = self.worker.policy(env, self)
        self._info.update(info)

        return env_action

    def on_step(self, env: EnvRun) -> None:
        logger.debug("worker.on_step()")

        # 初期化前はskip
        if not self._is_reset:
            return

        # 相手の番のrewardも加算
        self._step_reward += env.step_rewards[self.player_index]

        # 終了ならon_step実行
        if env.done:
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0

    # ------------------------------------
    # render functions
    # ------------------------------------
    def set_render_mode(self, mode: Union[str, PlayRenderMode]) -> None:
        self._render.reset(mode, interval=-1)

    def render(self, env: EnvRun, **kwargs) -> Union[None, str, np.ndarray]:
        # 初期化前はskip
        if not self._is_reset:
            return self._render.get_dummy()

        return self._render.render(env=env, worker=self, **kwargs)

    def render_terminal(self, env: EnvRun, return_text: bool = False, **kwargs):
        # 初期化前はskip
        if not self._is_reset:
            if return_text:
                return ""
            return

        return self._render.render_terminal(return_text, env=env, worker=self, **kwargs)

    def render_rgb_array(self, env: EnvRun, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_rgb_array(env=env, worker=self, **kwargs)

    def render_window(self, env: EnvRun, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_window(env=env, worker=self, **kwargs)
