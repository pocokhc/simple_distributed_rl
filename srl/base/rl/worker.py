import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    InfoType,
    InvalidActionsType,
    InvalidActionType,
    RLActionType,
    RLObservationType,
    RLTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.render import IRender
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig

if TYPE_CHECKING:
    from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


class WorkerBase(ABC, IRender):
    def __init__(self, training: bool, distributed: bool):
        self.__training = training
        self.__distributed = distributed

    # ------------------------------
    # implement
    # ------------------------------
    @property
    @abstractmethod
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    # ------------------------------
    # implement(option)
    # ------------------------------
    def render_terminal(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # ------------------------------------
    # run properties
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self.__training

    @property
    def distributed(self) -> bool:
        return self.__distributed


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
        # self.__env = None
        self.__recent_states = [self.__dummy_state for _ in range(self.config.window_length)]

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(self, state: EnvObservationType, env: EnvRun) -> RLObservationType:
        for processor in self.config.run_processors:
            state = processor.process_observation(state, env)

        if self.config.observation_type == RLTypes.DISCRETE:
            state = self.config.observation_space.encode_to_int_np(state)
        elif self.config.observation_type == RLTypes.CONTINUOUS:
            state = self.config.observation_space.encode_to_np(state)
        else:
            state = np.asarray(state)
        if state.shape == ():
            state = state.reshape((1,))
        return state

    def action_encode(self, action: EnvActionType) -> RLActionType:
        if self.config.action_type == RLTypes.DISCRETE:
            action = self.config.action_space.encode_to_int(action)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            action = self.config.action_space.encode_to_list_float(action)
        else:
            pass  # do nothing
        return action  # type: ignore

    def action_decode(self, action: RLActionType) -> EnvActionType:
        if self.config.action_type == RLTypes.DISCRETE:
            assert not isinstance(action, list)
            action = self.config.action_space.decode_from_int(int(action))
        elif self.config.action_type == RLTypes.CONTINUOUS:
            if isinstance(action, list):
                action = [float(a) for a in action]
            else:
                action = [float(action)]
            action = self.config.action_space.decode_from_list_float(action)
        else:
            pass  # do nothing
        return action

    def reward_encode(self, reward: float, env: EnvRun) -> float:
        for processor in self.config.run_processors:
            reward = processor.process_reward(reward, env)
        return reward

    # ------------------------------
    # implement
    # ------------------------------
    @abstractmethod
    def _call_on_reset(self, state: RLObservationType, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def _call_policy(
        self, state: RLObservationType, env: EnvRun, worker: "WorkerRun"
    ) -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def _call_on_step(
        self,
        next_state: RLObservationType,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: "WorkerRun",
    ) -> InfoType:
        raise NotImplementedError()

    # ------------------------------------
    # episode
    # ------------------------------------
    @property
    def player_index(self) -> int:
        return self.__player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self.__recent_states = [self.__dummy_state for _ in range(self.config.window_length)]
        self.__player_index = worker.player_index
        self.__env = env

        state = self.state_encode(env.state, env)
        self.__recent_states.pop(0)
        self.__recent_states.append(state)

        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)

        state = state.astype(np.float32)
        return self._call_on_reset(state, env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        # stacked state
        if self.config.window_length > 1:
            state = np.asarray(self.__recent_states)
        else:
            state = self.__recent_states[-1]

        state = state.astype(np.float32)
        action, info = self._call_policy(state, env, worker)
        action = self.action_decode(action)
        return action, info

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        next_state = self.state_encode(env.state, env)
        reward = self.reward_encode(worker.reward, env)

        self.__recent_states.pop(0)
        self.__recent_states.append(next_state)

        # stacked state
        if self.config.window_length > 1:
            next_state = np.asarray(self.__recent_states)

        next_state = next_state.astype(np.float32)
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
        return self.__env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.__env.player_num

    @property
    def step(self) -> int:
        return self.__env.step_num

    @property
    def recent_states(self) -> List[np.ndarray]:
        return self.__recent_states

    # ------------------------------------
    # utils
    # ------------------------------------
    def get_invalid_actions(self, env=None) -> InvalidActionsType:
        if self.config.action_type == RLTypes.DISCRETE:
            if env is None:
                env = self.__env
            return [cast(InvalidActionType, self.action_encode(a)) for a in env.get_invalid_actions(self.player_index)]
        else:
            return []

    def get_valid_actions(self, env=None) -> InvalidActionsType:
        if self.config.action_type == RLTypes.DISCRETE:
            if env is None:
                env = self.__env
            invalid_actions = self.get_invalid_actions(env)
            return [a for a in range(env.action_space.n) if a not in invalid_actions]
        else:
            return []

    def sample_action(self, env=None) -> RLActionType:
        if self.config.action_type == RLTypes.DISCRETE:
            action = np.random.choice(self.get_valid_actions(env))
        else:
            if env is None:
                env = self.__env
            action = env.sample(self.player_index)
            action = self.action_encode(action)
        return action

    def env_step(self, env: EnvRun, action: RLActionType, **kwargs) -> Tuple[np.ndarray, List[float], bool]:
        """Advance env one step

        Args:
            env (EnvRun): env
            action (RLAction): action

        Returns:
            Tuple[np.ndarray, List[float], bool]: 次の状態, 報酬, 終了したかどうか
        """
        env_action = self.action_decode(action)
        env.step(env_action, **kwargs)
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
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self._player_index = worker.player_index
        return self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return self.call_on_step(env, worker)


class ExtendWorker(WorkerBase):
    def __init__(
        self,
        rl_worker: "WorkerRun",
        training: bool,
        distributed: bool,
    ):
        super().__init__(training, distributed)
        self.rl_worker = rl_worker
        self.worker = self.rl_worker.worker

    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self._player_index = worker.player_index
        self.rl_worker.on_reset(env, worker.player_index)
        return self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self.rl_worker.on_step(env)
        return self.call_on_step(env, worker)
