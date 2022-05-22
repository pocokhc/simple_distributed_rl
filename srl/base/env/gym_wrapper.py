import pickle
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import EnvAction, EnvInvalidAction, EnvObservationType, Info
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace

from .base import EnvBase, SpaceBase

try:
    import gym
    from gym import spaces

except ImportError:
    pass


class GymWrapper(EnvBase):
    def __init__(self, env_name: str, prediction_by_simulation: bool):

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
            return

        if isinstance(space, spaces.Tuple):
            # すべてDiscreteならdiscrete
            if self._is_tuple_all_discrete(space):
                nvec = [s.n for s in space.spaces]
                self._action_space = ArrayDiscreteSpace(nvec)
                return
            else:
                raise ValueError  # TODO

        if isinstance(space, spaces.Box):
            self._action_space = BoxSpace(space.shape, space.low, space.high)

    def _pred_observation_space(self, space):
        if isinstance(space, spaces.Discrete):
            self._observation_space = DiscreteSpace(space.n)
            self._observation_type = EnvObservationType.DISCRETE
            return

        if isinstance(space, spaces.Tuple):
            # すべてDiscreteならdiscrete
            if self._is_tuple_all_discrete(space):
                nvec = [s.n for s in space.spaces]
                self._observation_space = ArrayDiscreteSpace(nvec)
                self._observation_type = EnvObservationType.DISCRETE
                return
            else:
                raise ValueError  # TODO

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

    # -----------------------------

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

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state = self.env.reset()
        return np.asarray(state), [0]

    def step(self, actions: List[EnvAction]) -> Tuple[np.ndarray, List[float], bool, List[int], Info]:
        state, reward, done, info = self.env.step(actions[0])
        return np.asarray(state), [float(reward)], done, [0], info

    def get_next_player_indices(self) -> List[int]:
        return [0]

    def get_invalid_actions(self, player_index: int) -> List[EnvInvalidAction]:
        return []

    def render_terminal(self) -> None:
        if "ansi" in self.render_modes:
            print(self.env.render("ansi"))

    def render_gui(self) -> None:
        if "human" in self.render_modes:
            self.env.render("human")

    def render_rgb_array(self) -> np.ndarray:
        if "rgb_array" in self.render_modes:
            return np.asarray(self.env.render("rgb_array"))
        else:
            raise NotImplementedError

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def get_original_env(self) -> object:
        return self.env
