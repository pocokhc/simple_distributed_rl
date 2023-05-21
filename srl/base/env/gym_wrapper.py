import logging
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym.spaces import flatten, flatten_space

from srl.base.define import EnvActionType, EnvObservationTypes, InfoType, RenderModes
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.utils.common import compare_less_version, is_package_installed

logger = logging.getLogger(__name__)


# v0.26.0 から大幅に変更
# https://github.com/openai/gym/releases

"""
gym_spaceを1次元にして管理する
decodeもあるので順序を保持する
"""


def _gym_space_flatten_sub(gym_space) -> Tuple[List[float], List[float], bool]:
    if isinstance(gym_space, spaces.Discrete):
        if hasattr(gym_space, "start"):
            return [gym_space.start], [gym_space.start + gym_space.n - 1], True
        else:
            return [0], [gym_space.n - 1], True

    if isinstance(gym_space, spaces.MultiDiscrete):
        nvec = gym_space.nvec.flatten()
        return np.zeros(nvec.shape).tolist(), nvec.tolist(), True

    if isinstance(gym_space, spaces.MultiBinary):
        shape = np.zeros(gym_space.shape).flatten().shape
        return np.zeros(shape).tolist(), np.ones(shape).tolist(), True

    if isinstance(gym_space, spaces.Box):
        return gym_space.low.flatten().tolist(), gym_space.high.flatten().tolist(), False

    if isinstance(gym_space, spaces.Tuple):
        low = []
        high = []
        is_discrete = True
        for c in gym_space.spaces:
            _l, _h, _is_d = _gym_space_flatten_sub(c)
            low.extend(_l)
            high.extend(_h)
            if not _is_d:
                is_discrete = False
        return low, high, is_discrete

    if isinstance(gym_space, spaces.Dict):
        low = []
        high = []
        is_discrete = True
        for k in sorted(gym_space.spaces.keys()):
            space = gym_space.spaces[k]
            _l, _h, _is_d = _gym_space_flatten_sub(space)
            low.extend(_l)
            high.extend(_h)
            if not _is_d:
                is_discrete = False
        return low, high, is_discrete

    if isinstance(gym_space, spaces.Graph):
        pass  # TODO

    if isinstance(gym_space, spaces.Text):
        shape = (gym_space.max_length,)
        return np.zeros(shape).tolist(), np.full(shape, len(gym_space.character_set)).tolist(), True

    if isinstance(gym_space, spaces.Sequence):
        pass  # TODO

    # other space
    flat_space = flatten_space(gym_space)
    if isinstance(flat_space, spaces.Box):
        return flat_space.low.tolist(), flat_space.high.tolist(), False

    raise NotImplementedError(f"not supported `{gym_space}`")


def gym_space_flatten(gym_space) -> Tuple[Union[BoxSpace, ArrayDiscreteSpace], bool]:
    low, high, is_discrete = _gym_space_flatten_sub(gym_space)
    if is_discrete:
        low = [int(n) for n in low]
        high = [int(n) for n in high]
        return ArrayDiscreteSpace(len(low), low, high), is_discrete
    else:
        return BoxSpace((len(low),), low, high), is_discrete


def _gym_space_flatten_encode_sub(gym_space, x):
    if isinstance(gym_space, spaces.Discrete):
        return np.array([x])

    if isinstance(gym_space, spaces.MultiDiscrete):
        return x.flatten()

    if isinstance(gym_space, spaces.MultiBinary):
        return x.flatten()

    if isinstance(gym_space, spaces.Box):
        return x.flatten()

    if isinstance(gym_space, spaces.Tuple):
        return np.concatenate(
            [_gym_space_flatten_encode_sub(space, x_part) for space, x_part in zip(gym_space.spaces, x)],
        )

    if isinstance(gym_space, spaces.Dict):
        keys = sorted(gym_space.spaces.keys())
        return np.concatenate(
            [_gym_space_flatten_encode_sub(gym_space.spaces[key], x[key]) for key in keys],
        )

    if isinstance(gym_space, spaces.Graph):
        pass  # TODO

    if isinstance(gym_space, spaces.Text):
        arr = np.full(shape=(gym_space.max_length,), fill_value=len(gym_space.character_set), dtype=np.int32)
        for i, val in enumerate(x):
            arr[i] = gym_space.character_index(val)
        return arr

    if isinstance(gym_space, spaces.Sequence):
        pass  # TODO

    # other space
    x = flatten(gym_space, x)
    if isinstance(x, np.ndarray):
        return x

    raise NotImplementedError(f"not supported `{gym_space}` `{x}`")


def gym_space_flatten_encode(gym_space, val):
    # 主に状態
    return _gym_space_flatten_encode_sub(gym_space, val)


def _gym_space_flatten_decode_sub(gym_space: spaces.Space, x, idx=0):
    if isinstance(gym_space, spaces.Discrete):
        return int(x[idx]), idx + 1

    if isinstance(gym_space, spaces.MultiDiscrete):
        size = len(gym_space.nvec.flatten())
        arr = x[idx : idx + size]
        return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, spaces.MultiBinary):
        size = len(np.zeros(gym_space.shape).flatten())
        arr = x[idx : idx + size]
        return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, spaces.Box):
        if gym_space.shape == ():
            size = 1
            arr = x[idx : idx + size]
            return np.asarray(arr).astype(gym_space.dtype), idx + size
        else:
            size = len(np.zeros(gym_space.shape).flatten())
            arr = x[idx : idx + size]
            return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, spaces.Tuple):
        arr = []
        for space in gym_space.spaces:
            n, idx = _gym_space_flatten_decode_sub(space, x, idx)
            arr.append(n)
        return tuple(arr), idx

    if isinstance(gym_space, spaces.Dict):
        keys = sorted(gym_space.spaces.keys())
        dic = {}
        for key in keys:
            n, idx = _gym_space_flatten_decode_sub(gym_space.spaces[key], x, idx)
            dic[key] = n
        return dic, idx

    if isinstance(gym_space, spaces.Graph):
        pass  # TODO

    if isinstance(gym_space, spaces.Text):
        pass  # TODO

    if isinstance(gym_space, spaces.Sequence):
        pass  # TODO

    raise NotImplementedError(f"not supported `{gym_space}` `{x}`")


def gym_space_flatten_decode(gym_space: spaces.Space, val) -> Any:
    # 主にアクション
    if isinstance(val, tuple):
        val = list(val)
    elif isinstance(val, np.ndarray):
        pass
    elif not isinstance(val, list):
        val = [val]
    val, _ = _gym_space_flatten_decode_sub(gym_space, val)
    return val


class GymWrapper(EnvBase):
    def __init__(
        self,
        env_name: str,
        arguments: dict,
        check_image: bool = True,
        prediction_by_simulation: bool = True,
        prediction_step: int = 10,
    ):
        self.prediction_step = prediction_step

        self.seed = None
        self.render_mode = RenderModes.NONE
        self.v0260_older = compare_less_version(gym.__version__, "0.26.0")  # type: ignore
        if False:
            if is_package_installed("ale_py"):
                import ale_py

                if self.v0260_older:
                    assert compare_less_version(ale_py.__version__, "0.8.0")
                else:
                    assert not compare_less_version(ale_py.__version__, "0.8.0")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        logger.info("set SDL_VIDEODRIVER='dummy'")

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

        # --- space img
        self._observation_type = EnvObservationTypes.UNKNOWN
        if check_image:
            if isinstance(self.env.observation_space, spaces.Box) and (
                "uint" in str(self.env.observation_space.dtype)
            ):
                if len(self.env.observation_space.shape) == 2:
                    self._observation_type = EnvObservationTypes.GRAY_2ch
                elif len(self.env.observation_space.shape) == 3:
                    # w,h,ch 想定
                    ch = self.env.observation_space.shape[-1]
                    if ch == 1:
                        self._observation_type = EnvObservationTypes.GRAY_3ch
                    elif ch == 3:
                        self._observation_type = EnvObservationTypes.COLOR

                if self._observation_type != EnvObservationTypes.UNKNOWN:
                    # 画像はそのままのshape
                    self.enable_flatten_observation = False
                    self._observation_space = BoxSpace(
                        self.env.observation_space.shape,
                        self.env.observation_space.low,
                        self.env.observation_space.high,
                    )

        # --- space obs
        if self._observation_type == EnvObservationTypes.UNKNOWN:
            self.enable_flatten_observation = True
            self._observation_space, is_discrete = gym_space_flatten(self.env.observation_space)
            if not is_discrete:
                if self._pred_space_discrete():
                    self._observation_type = EnvObservationTypes.DISCRETE
                else:
                    self._observation_type = EnvObservationTypes.CONTINUOUS
            else:
                self._observation_type = EnvObservationTypes.DISCRETE

        # --- space action
        self.enable_flatten_action = True
        self._action_space, _ = gym_space_flatten(self.env.action_space)

        logger.info(f"obs original: {self.env.observation_space}")
        logger.info(f"act original: {self.env.action_space}")
        logger.info(f"obs_type   : {self._observation_type}")
        logger.info(f"observation: {self._observation_space}")
        logger.info(f"flatten_obs: {self.enable_flatten_observation}")
        logger.info(f"action     : {self._action_space}")
        logger.info(f"flatten_act: {self.enable_flatten_action}")

    def _pred_space_discrete(self):
        # 実際に値を取得して予測
        done = True
        for _ in range(self.prediction_step):
            if done:
                state = self.env.reset()
                if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                    state, _ = state
                done = False
            else:
                action = self.env.action_space.sample()
                _t = self.env.step(action)
                if len(_t) == 4:
                    state, reward, done, info = _t  # type: ignore
                else:
                    state, reward, terminated, truncated, info = _t
                    done = terminated or truncated
            flat_state = flatten(self.env.observation_space, state)
            if "int" not in str(np.asarray(flat_state).dtype):
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
    def observation_type(self) -> EnvObservationTypes:
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

    @property
    def next_player_index(self) -> int:
        return 0

    def reset(self) -> Tuple[np.ndarray, dict]:
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
        if self.enable_flatten_observation:
            state = gym_space_flatten_encode(self.env.observation_space, state)
        return self.observation_space.convert(state), info

    def step(self, action: EnvActionType) -> Tuple[np.ndarray, List[float], bool, InfoType]:
        if self.enable_flatten_action:
            action = gym_space_flatten_decode(self.env.action_space, action)
        _t = self.env.step(action)
        if len(_t) == 4:
            state, reward, done, info = _t  # type: ignore
        else:
            state, reward, terminated, truncated, info = _t
            done = terminated or truncated
        if self.enable_flatten_observation:
            state = gym_space_flatten_encode(self.env.observation_space, state)
        return self.observation_space.convert(state), [float(reward)], done, info

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def close(self) -> None:
        self.env.close()
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

    def set_render_mode(self, mode: RenderModes) -> None:
        if self.v0260_older:
            return

        # modeが違っていたら作り直す
        if mode == RenderModes.Terminal:
            if self.render_mode != RenderModes.Terminal and "ansi" in self.render_modes:
                self.env = gym.make(self.name, render_mode="ansi", **self.arguments)
                self.render_mode = RenderModes.Terminal
        elif mode == RenderModes.RBG_array:
            if self.render_mode != RenderModes.RBG_array and "rgb_array" in self.render_modes:
                self.env = gym.make(self.name, render_mode="rgb_array", **self.arguments)
                self.render_mode = RenderModes.RBG_array

    def render_terminal(self, **kwargs) -> None:
        if self.v0260_older:
            if "ansi" in self.render_modes:
                print(self.env.render(mode="ansi", **kwargs))  # type: ignore
        else:
            if self.render_mode == RenderModes.Terminal:
                print(self.env.render(**kwargs))

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if self.v0260_older:
            if "rgb_array" in self.render_modes:
                return np.asarray(self.env.render(mode="rgb_array", **kwargs))  # type: ignore
        else:
            if self.render_mode == RenderModes.RBG_array:
                return np.asarray(self.env.render(**kwargs))
        return None
