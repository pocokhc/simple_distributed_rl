import logging
import os
import pickle
from typing import Any, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from srl.base.define import EnvAction, EnvObservationType, Info, RenderMode
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.utils.common import compare_less_version, is_package_installed

logger = logging.getLogger(__name__)


# v0.26.0 から大幅に変更
# https://github.com/openai/gym/releases


class GymWrapper(EnvBase):
    def __init__(self, env_name: str, arguments: dict, prediction_by_simulation: bool):
        self.seed = None
        self.render_mode = RenderMode.NONE
        self.v0260_older = compare_less_version(gym.__version__, "0.26.0")
        if False:
            if is_package_installed("ale_py"):
                import ale_py

                if self.v0260_older:
                    assert compare_less_version(ale_py.__version__, "0.8.0")
                else:
                    assert not compare_less_version(ale_py.__version__, "0.8.0")
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.name = env_name
        self.arguments = arguments
        self.env: gym.Env = gym.make(env_name, **self.arguments)
        logger.info(f"metadata: {self.env.metadata}")

        # fps
        self.fps = self.env.metadata.get("render_fps", 60)

        # render_modes
        self.render_modes = ["ansi", "human", "rgb_array"]
        if "render.modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render.modes"]
        elif "render_modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render_modes"]

        self.prediction_by_simulation = prediction_by_simulation

        self._observation_type = EnvObservationType.UNKNOWN
        self._action_space = self._pred_space(self.env.action_space, True)
        self._observation_space, self._observation_type = self._pred_space(self.env.observation_space, False)

    def _pred_space(self, gym_space, is_action: bool):
        if isinstance(gym_space, spaces.Discrete):
            space = DiscreteSpace(gym_space.n)
            logger.debug(f"space: {space}")
            if is_action:
                return space
            else:
                return space, EnvObservationType.DISCRETE

        if isinstance(gym_space, spaces.Tuple):
            # すべてDiscreteならdiscrete
            if self._is_tuple_all_discrete(gym_space):
                high = [s.n - 1 for s in gym_space.spaces]
                space = ArrayDiscreteSpace(len(high), 0, high)
                logger.debug(f"space: {space}")
                if is_action:
                    return space
                else:
                    return space, EnvObservationType.DISCRETE
            else:
                pass  # TODO

        if isinstance(gym_space, spaces.Box):
            if is_action:
                return BoxSpace(gym_space.shape, gym_space.low, gym_space.high)

            # 状態は、離散の可能性を確認
            space_type = EnvObservationType.UNKNOWN
            if self._observation_type == EnvObservationType.UNKNOWN and len(gym_space.shape) == 1:
                if "int" in str(gym_space.dtype):
                    space_type = EnvObservationType.DISCRETE
                elif self._pred_space_discrete():
                    space_type = EnvObservationType.DISCRETE
                else:
                    space_type = EnvObservationType.CONTINUOUS

            if space_type == EnvObservationType.DISCRETE:
                if gym_space.shape[0] == 1:
                    return DiscreteSpace(gym_space.high[0]), space_type
                else:
                    return BoxSpace(gym_space.shape, gym_space.low, gym_space.high), space_type
            else:
                return BoxSpace(gym_space.shape, gym_space.low, gym_space.high), space_type

        raise ValueError(f"not supported({gym_space})")

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
                if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                    state, _ = state
                done = False
            else:
                action = self.env.action_space.sample()
                _t = self.env.step(action)
                if len(_t) == 4:
                    state, reward, done, info = _t
                else:
                    state, reward, terminated, truncated, info = _t
                    done = terminated or truncated
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

    def reset(self) -> Tuple[np.ndarray, int, dict]:
        if self.seed is None:
            state = self.env.reset()
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                state, info = state
            else:
                info = {}
        else:
            # seed を最初のみ設定
            state = self.env.reset(seed=self.seed)
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                state, info = state
            else:
                info = {}
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
            self.seed = None
        return self.observation_space.convert(state), 0, info

    def step(
        self,
        action: EnvAction,
        player_index: int,
    ) -> Tuple[np.ndarray, List[float], bool, int, Info]:
        _t = self.env.step(action)
        if len(_t) == 4:
            state, reward, done, info = _t
        else:
            state, reward, terminated, truncated, info = _t
            done = terminated or truncated
        return self.observation_space.convert(state), [float(reward)], done, 0, info

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def close(self) -> None:
        # self.env.close()
        # render 内で使われている pygame に対して close -> init をするとエラーになる
        # Fatal Python error: (pygame parachute) Segmentation Fault
        pass

    def get_original_env(self) -> object:
        return self.env

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    @property
    def render_interval(self) -> float:
        return 1000 / self.fps

    def set_render_mode(self, mode: RenderMode) -> None:
        if self.v0260_older:
            return

        # modeが違っていたら作り直す
        if mode == RenderMode.Terminal:
            if self.render_mode != RenderMode.Terminal and "ansi" in self.render_modes:
                self.env = gym.make(self.name, render_mode="ansi", **self.arguments)
                self.render_mode = RenderMode.Terminal
        elif mode == RenderMode.RBG_array:
            if self.render_mode != RenderMode.RBG_array and "rgb_array" in self.render_modes:
                self.env = gym.make(self.name, render_mode="rgb_array", **self.arguments)
                self.render_mode = RenderMode.RBG_array

    def render_terminal(self, **kwargs) -> None:
        if self.v0260_older:
            if "ansi" in self.render_modes:
                print(self.env.render(mode="ansi", **kwargs))
        else:
            if self.render_mode == RenderMode.Terminal:
                print(self.env.render(**kwargs))

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if self.v0260_older:
            if "rgb_array" in self.render_modes:
                return np.asarray(self.env.render(mode="rgb_array", **kwargs))
        else:
            if self.render_mode == RenderMode.RBG_array:
                return np.asarray(self.env.render(**kwargs))
        return None
