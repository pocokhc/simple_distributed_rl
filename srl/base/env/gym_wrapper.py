import logging
import os
import pickle
from typing import Any, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from gym import spaces as gym_spaces
from gym.spaces import flatten, flatten_space

from srl.base.define import EnvActionType, EnvObservationTypes, InfoType, RenderModes
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.config import EnvConfig
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.utils.common import compare_less_version, is_package_installed

logger = logging.getLogger(__name__)


# v0.26.0 から大幅に変更
# https://github.com/openai/gym/releases

"""
・gym_spaceを1次元にして管理する
・decodeもあるので順序を保持する
・変換できないものはエラーログを出力して無視する
"""


def _gym_space_flatten_sub(
    gym_space: gym_spaces.Space,
) -> Tuple[List[Union[int, float]], List[Union[int, float]], bool]:
    """
    Returns:
        Tuple[
            List[float]: low,
            List[float]: high,
            bool       : is_discrete,
        ]
    """
    if isinstance(gym_space, gym_spaces.Discrete):
        if hasattr(gym_space, "start"):
            return [int(gym_space.start)], [int(gym_space.start + gym_space.n - 1)], True
        else:
            return [0], [int(gym_space.n - 1)], True

    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        nvec = gym_space.nvec.flatten()
        return np.zeros(nvec.shape).tolist(), nvec.tolist(), True

    if isinstance(gym_space, gym_spaces.MultiBinary):
        shape = np.zeros(gym_space.shape).flatten().shape
        return np.zeros(shape).tolist(), np.ones(shape).tolist(), True

    if isinstance(gym_space, gym_spaces.Box):
        return gym_space.low.flatten().tolist(), gym_space.high.flatten().tolist(), False

    if isinstance(gym_space, gym_spaces.Tuple):
        low = []
        high = []
        is_discrete = True
        for c in gym_space.spaces:
            _l, _h, _is_d = _gym_space_flatten_sub(c)
            if len(_l) > 0:
                low.extend(_l)
                high.extend(_h)
                if not _is_d:
                    is_discrete = False
        return low, high, is_discrete

    if isinstance(gym_space, gym_spaces.Dict):
        low = []
        high = []
        is_discrete = True
        for k in sorted(gym_space.spaces.keys()):
            space = gym_space.spaces[k]
            _l, _h, _is_d = _gym_space_flatten_sub(space)
            if len(_l) > 0:
                low.extend(_l)
                high.extend(_h)
                if not _is_d:
                    is_discrete = False
        return low, high, is_discrete

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # TODO

    if hasattr(gym_spaces, "Text") and isinstance(gym_space, gym_spaces.Text):
        shape = (gym_space.max_length,)
        return np.zeros(shape).tolist(), np.full(shape, len(gym_space.character_set)).tolist(), True

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # TODO

    # ---- other space
    try:
        flat_space = flatten_space(gym_space)
        if isinstance(flat_space, gym_spaces.Box):
            return flat_space.low.tolist(), flat_space.high.tolist(), False
    except NotImplementedError as e:
        logger.warning(f"Ignored for unsupported space. type '{type(gym_space)}', err_msg '{e}'")

    return [], [], False


def gym_space_flatten(gym_space: gym_spaces.Space) -> Tuple[Union[BoxSpace, ArrayDiscreteSpace], bool]:
    low, high, is_discrete = _gym_space_flatten_sub(gym_space)
    assert len(low) > 0, "Space flatten failed."
    assert len(low) == len(high)
    if is_discrete:
        low = [int(n) for n in low]
        high = [int(n) for n in high]
        return ArrayDiscreteSpace(len(low), low, high), is_discrete
    else:
        return BoxSpace((len(low),), low, high), is_discrete


def _gym_space_flatten_encode_sub(gym_space: gym_spaces.Space, x: Any):
    if isinstance(gym_space, gym_spaces.Discrete):
        return np.array([x])

    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        return x.flatten()

    if isinstance(gym_space, gym_spaces.MultiBinary):
        return x.flatten()

    if isinstance(gym_space, gym_spaces.Box):
        return x.flatten()

    if isinstance(gym_space, gym_spaces.Tuple):
        x = cast(Any, x)
        s = cast(Any, gym_space.spaces)
        x = [_gym_space_flatten_encode_sub(space, x_part) for space, x_part in zip(s, x)]
        return np.concatenate([x for x in x if x is not None])

    if isinstance(gym_space, gym_spaces.Dict):
        keys = sorted(gym_space.spaces.keys())
        x = [_gym_space_flatten_encode_sub(gym_space.spaces[key], x[key]) for key in keys]
        return np.concatenate([x for x in x if x is not None])

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # TODO

    if hasattr(gym_spaces, "Text") and isinstance(gym_space, gym_spaces.Text):
        arr = np.full(shape=(gym_space.max_length,), fill_value=len(gym_space.character_set), dtype=np.int32)
        for i, val in enumerate(x):
            arr[i] = gym_space.character_index(val)
        return arr

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # TODO

    # ---- other space
    try:
        x = flatten(gym_space, x)
        if isinstance(x, np.ndarray):
            return x
    except NotImplementedError as e:
        logger.debug(f"Ignored for unsupported space. type '{type(gym_space)}', '{x}', err_msg '{e}'")

    return None


def gym_space_flatten_encode(gym_space: gym_spaces.Space, val: Any):
    x = _gym_space_flatten_encode_sub(gym_space, val)
    assert x is not None, "Space flatten encode failed."
    return x


def _gym_space_flatten_decode_sub(gym_space: gym_spaces.Space, x: Any, idx: int = 0):
    if isinstance(gym_space, gym_spaces.Discrete):
        return int(x[idx]), idx + 1

    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        size = len(gym_space.nvec.flatten())
        arr = x[idx : idx + size]
        return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, gym_spaces.MultiBinary):
        size = len(np.zeros(gym_space.shape).flatten())
        arr = x[idx : idx + size]
        return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, gym_spaces.Box):
        if gym_space.shape == ():
            size = 1
            arr = x[idx : idx + size]
            return np.asarray(arr).astype(gym_space.dtype), idx + size
        else:
            size = len(np.zeros(gym_space.shape).flatten())
            arr = x[idx : idx + size]
            return np.asarray(arr).reshape(gym_space.shape).astype(gym_space.dtype), idx + size

    if isinstance(gym_space, gym_spaces.Tuple):
        arr = []
        for space in gym_space.spaces:
            n, idx = _gym_space_flatten_decode_sub(space, x, idx)
            arr.append(n)
        return tuple(arr), idx

    if isinstance(gym_space, gym_spaces.Dict):
        keys = sorted(gym_space.spaces.keys())
        dic = {}
        for key in keys:
            n, idx = _gym_space_flatten_decode_sub(gym_space.spaces[key], x, idx)
            dic[key] = n
        return dic, idx

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # TODO

    # if isinstance(gym_space, gym_spaces.Text):
    #    pass  # TODO

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # TODO

    # 不明なのはsampleがあればそれを適用、なければNone
    if hasattr(gym_space, "sample"):
        y = gym_space.sample()
    else:
        y = None
    return y, idx


def gym_space_flatten_decode(gym_space: gym_spaces.Space, val: Any) -> Any:
    if isinstance(val, tuple):
        val = list(val)
    elif isinstance(val, np.ndarray):
        pass
    elif not isinstance(val, list):
        val = [val]
    val, _ = _gym_space_flatten_decode_sub(gym_space, val)
    assert val is not None, "Space flatten decode failed."
    return val


class GymWrapper(EnvBase):
    def __init__(self, config: EnvConfig):
        self.config = config

        self.seed = None
        self.render_mode = RenderModes.none
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

        self.env = self.make_gym_env()
        logger.info(f"metadata    : {self.env.metadata}")
        logger.info(f"action_space: {self.env.action_space}")
        logger.info(f"obs_space   : {self.env.observation_space}")

        # fps
        self.fps = self.env.metadata.get("render_fps", 60)

        # render_modes
        self.render_modes = ["ansi", "human", "rgb_array"]
        if "render.modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render.modes"]
        elif "render_modes" in self.env.metadata:
            self.render_modes = self.env.metadata["render_modes"]

        _act_space = None
        _obs_type = EnvObservationTypes.UNKNOWN
        _obs_space = None
        self.enable_flatten_action = False
        self.enable_flatten_observation = False

        # --- wrapper
        for wrapper in config.gym_wrappers:
            _act_space = wrapper.action_space(_act_space, self.env)
            _obs_type, _obs_space = wrapper.observation_space(_obs_type, _obs_space, self.env)

        # --- space img
        if _obs_space is None:
            if config.gym_check_image:
                if isinstance(self.env.observation_space, gym_spaces.Box) and (
                    "uint" in str(self.env.observation_space.dtype)
                ):
                    if len(self.env.observation_space.shape) == 2:
                        _obs_type = EnvObservationTypes.GRAY_2ch
                    elif len(self.env.observation_space.shape) == 3:
                        # w,h,ch 想定
                        ch = self.env.observation_space.shape[-1]
                        if ch == 1:
                            _obs_type = EnvObservationTypes.GRAY_3ch
                        elif ch == 3:
                            _obs_type = EnvObservationTypes.COLOR

                    if _obs_type != EnvObservationTypes.UNKNOWN:
                        # 画像はそのままのshape
                        self.enable_flatten_observation = False
                        _obs_space = BoxSpace(
                            self.env.observation_space.shape,
                            self.env.observation_space.low,
                            self.env.observation_space.high,
                        )

            # --- space obs
            if _obs_type == EnvObservationTypes.UNKNOWN:
                self.enable_flatten_observation = True
                _obs_space, is_discrete = gym_space_flatten(self.env.observation_space)
                if not is_discrete:
                    if self.config.gym_prediction_by_simulation and self._pred_space_discrete():
                        _obs_type = EnvObservationTypes.DISCRETE
                    else:
                        _obs_type = EnvObservationTypes.CONTINUOUS
                else:
                    _obs_type = EnvObservationTypes.DISCRETE

        # --- space action
        if _act_space is None:
            self.enable_flatten_action = True
            _act_space, _ = gym_space_flatten(self.env.action_space)

        assert _obs_space is not None
        self._action_space: SpaceBase = _act_space
        self._observation_type = _obs_type
        self._observation_space: SpaceBase = _obs_space

        logger.info(f"obs_type   : {self._observation_type}")
        logger.info(f"observation: {self._observation_space}")
        logger.info(f"flatten_obs: {self.enable_flatten_observation}")
        logger.info(f"action     : {self._action_space}")
        logger.info(f"flatten_act: {self.enable_flatten_action}")

    def make_gym_env(self, **kwargs) -> gym.Env:
        if self.config.gym_make_func is None:
            return gym.make(self.config.name, **self.config.kwargs, **kwargs)
        return self.config.gym_make_func(self.config.name, **self.config.kwargs, **kwargs)

    def _pred_space_discrete(self):
        # 実際に値を取得して予測
        done = True
        for _ in range(self.config.gym_prediction_step):
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
        elif hasattr(self.env, "spec"):
            if hasattr(self.env.spec, "max_episode_steps"):
                if self.env.spec.max_episode_steps is not None:
                    return self.env.spec.max_episode_steps
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

        # wrapper
        for w in self.config.gym_wrappers:
            state = w.observation(state, self.env)

        # flatten
        if self.enable_flatten_observation:
            state = gym_space_flatten_encode(self.env.observation_space, state)

        state = self.observation_space.convert(state)
        return state, info

    def step(self, action: EnvActionType) -> Tuple[np.ndarray, List[float], bool, InfoType]:
        # wrapper
        for w in self.config.gym_wrappers:
            action = w.action(action, self.env)

        # flatten
        if self.enable_flatten_action:
            action = gym_space_flatten_decode(self.env.action_space, action)

        # step
        _t = self.env.step(action)
        if len(_t) == 4:
            state, reward, done, info = _t  # type: ignore
        else:
            state, reward, terminated, truncated, info = _t
            done = terminated or truncated

        # wrapper
        for w in self.config.gym_wrappers:
            state = w.observation(state, self.env)
            reward = w.reward(reward, self.env)
            done = w.done(done, self.env)

        # flatten
        if self.enable_flatten_observation:
            state = gym_space_flatten_encode(self.env.observation_space, state)

        state = self.observation_space.convert(state)
        return state, [float(reward)], done, info

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

    def set_render_terminal_mode(self) -> None:
        if self.v0260_older:
            return

        # modeが違っていたら作り直す
        if self.render_mode != RenderModes.terminal and "ansi" in self.render_modes:
            self.env = self.make_gym_env(render_mode="ansi")
            self.render_mode = RenderModes.terminal

    def set_render_rgb_mode(self) -> None:
        if self.v0260_older:
            return

        # modeが違っていたら作り直す
        if self.render_mode != RenderModes.rgb_array and "rgb_array" in self.render_modes:
            self.env = self.make_gym_env(render_mode="rgb_array")
            self.render_mode = RenderModes.rgb_array

    def render_terminal(self, **kwargs) -> None:
        if self.v0260_older:
            if "ansi" in self.render_modes:
                print(self.env.render(mode="ansi", **kwargs))  # type: ignore
        else:
            if self.render_mode == RenderModes.terminal:
                print(self.env.render(**kwargs))

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if self.v0260_older:
            if "rgb_array" in self.render_modes:
                return np.asarray(self.env.render(mode="rgb_array", **kwargs))  # type: ignore
        else:
            if self.render_mode == RenderModes.rgb_array:
                return np.asarray(self.env.render(**kwargs))
        return None
