import logging
import pickle
from typing import Any, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from srl.base.define import EnvAction, EnvObservationType, Info
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace

logger = logging.getLogger(__name__)


class GymWrapper(EnvBase):
    def __init__(self, env_name: str, prediction_by_simulation: bool):
        self.seed = None

        self.env: gym.Env = gym.make(env_name)

        self.prediction_by_simulation = prediction_by_simulation

        self._observation_type = EnvObservationType.UNKNOWN
        self._pred_action_space(self.env.action_space)
        self._pred_observation_space(self.env.observation_space)

        self.render_modes = ["ansi", "human", "rgb_array"]
        if "render.modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render.modes"]
        elif "render_modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render_modes"]

    def _pred_action_space(self, space):
        if isinstance(space, spaces.Discrete):
            self._action_space = DiscreteSpace(space.n)
            logger.debug(f"action_space: {self.action_space}")
            return

        if isinstance(space, spaces.Tuple):
            # すべてDiscreteならdiscrete
            if self._is_tuple_all_discrete(space):
                nvec = [s.n for s in space.spaces]
                self._action_space = ArrayDiscreteSpace(nvec)
                logger.debug(f"action_space: {self.action_space}")
                return
            else:
                pass  # TODO

        if isinstance(space, spaces.Box):
            self._action_space = BoxSpace(space.shape, space.low, space.high)
            logger.debug(f"action_space: {self.action_space}")
            return

        raise ValueError(f"not supported({space})")

    def _pred_observation_space(self, space):
        if isinstance(space, spaces.Discrete):
            self._observation_space = DiscreteSpace(space.n)
            self._observation_type = EnvObservationType.DISCRETE
            logger.debug(f"observation_space: {self.observation_type} {self.observation_space}")
            return

        if isinstance(space, spaces.Tuple):
            # すべてDiscreteならdiscrete
            if self._is_tuple_all_discrete(space):
                nvec = [s.n for s in space.spaces]
                self._observation_space = ArrayDiscreteSpace(nvec)
                self._observation_type = EnvObservationType.DISCRETE
                logger.debug(f"observation_space: {self.observation_type} {self.observation_space}")
                return
            else:
                pass  # TODO

        if isinstance(space, spaces.Box):
            # 離散の可能性を確認
            if self._observation_type == EnvObservationType.UNKNOWN and len(space.shape) == 1:
                if "int" in str(space.dtype) or (self.prediction_by_simulation and self._pred_space_discrete()):
                    self._observation_type == EnvObservationType.DISCRETE
                    if space.shape[0] == 1:
                        self._observation_space = DiscreteSpace(space.high[0])
                    else:
                        self._observation_space = BoxSpace(space.shape, space.low, space.high)
                else:
                    self._observation_space = BoxSpace(space.shape, space.low, space.high)
                    self._observation_type = EnvObservationType.CONTINUOUS
            else:
                self._observation_space = BoxSpace(space.shape, space.low, space.high)
            logger.debug(f"observation_space: {self.observation_type} {self.observation_space}")
            return

        raise ValueError(f"not supported({space})")

    def _is_tuple_all_discrete(self, space) -> bool:
        for s in space.spaces:
            if not isinstance(s, spaces.Discrete):
                return False
        return True

    def _pred_space_discrete(self):

        # 実際に値を取得して予測
        done = True
        for _ in range(10):
            if done:
                state = self.env.reset()
                done = False
            else:
                action = self.env.action_space.sample()
                state, _, done, _ = self.env.step(action)
            if "int" not in str(np.asarray(state).dtype):
                return False

        return True

    # --------------------------------
    # implement
    # --------------------------------

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self._observation_type

    @property
    def max_episode_steps(self) -> int:
        if hasattr(self.env, "_max_episode_steps"):
            return getattr(self.env, "_max_episode_steps")
        elif hasattr(self.env, "spec") and self.env.spec.max_episode_steps is not None:
            return self.env.spec.max_episode_steps
        else:
            return 99_999

    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[np.ndarray, int]:
        if self.seed is None:
            state = self.env.reset()
        else:
            # seed を最初のみ設定
            state = self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
            self.seed = None

        return np.asarray(state), 0

    def step(
        self,
        action: EnvAction,
        player_index: int,
    ) -> Tuple[np.ndarray, List[float], bool, int, Info]:
        state, reward, done, info = self.env.step(action)
        return np.asarray(state), [float(reward)], done, 0, info

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def close(self) -> None:
        # self.env.close()
        # render 内で使われている pygame に対して close -> init をするとエラーになる
        # Fatal Python error: (pygame parachute) Segmentation Fault
        pass

    def render_terminal(self, **kwargs) -> None:
        if "ansi" in self.render_modes:
            print(self.env.render(mode="ansi", **kwargs))

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        if "rgb_array" in self.render_modes:
            return np.asarray(self.env.render(mode="rgb_array", **kwargs))
        else:
            raise NotImplementedError

    def get_original_env(self) -> object:
        return self.env

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
