import logging
import os
import pickle
from typing import Any, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from gym import spaces as gym_spaces
from gym.spaces import flatten, flatten_space

from srl.base import spaces as srl_spaces
from srl.base.define import DoneTypes, EnvActionType, EnvObservationTypes, InfoType, RenderModes, RLTypes
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.config import EnvConfig
from srl.utils.common import compare_less_version, is_package_installed

logger = logging.getLogger(__name__)


# v0.26.0 から大幅に変更
# https://github.com/openai/gym/releases


def _space_change_from_gym_to_srl_sub(gym_space: gym_spaces.Space) -> Optional[Union[SpaceBase, List[SpaceBase]]]:
    if isinstance(gym_space, gym_spaces.Discrete):
        if hasattr(gym_space, "start"):
            return srl_spaces.DiscreteSpace(int(gym_space.n), start=int(gym_space.start))
        else:
            return srl_spaces.DiscreteSpace(int(gym_space.n))

    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        return srl_spaces.BoxSpace(gym_space.shape, 0, gym_space.nvec, dtype=np.int64)

    if isinstance(gym_space, gym_spaces.MultiBinary):
        return srl_spaces.BoxSpace(gym_space.shape, 0, 1, dtype=np.int8)

    if isinstance(gym_space, gym_spaces.Box):
        return srl_spaces.BoxSpace(gym_space.shape, gym_space.low, gym_space.high, gym_space.dtype)

    if isinstance(gym_space, gym_spaces.Tuple):
        sub_spaces = []
        for c in gym_space.spaces:
            sub_space = _space_change_from_gym_to_srl_sub(c)
            if sub_space is None:
                continue
            if isinstance(sub_space, list):
                sub_spaces.extend(sub_space)
            else:
                sub_spaces.append(sub_space)
        return sub_spaces

    if isinstance(gym_space, gym_spaces.Dict):
        sub_spaces = []
        for k in sorted(gym_space.spaces.keys()):
            space = gym_space.spaces[k]
            sub_space = _space_change_from_gym_to_srl_sub(space)
            if sub_space is None:
                continue
            if isinstance(sub_space, list):
                sub_spaces.extend(sub_space)
            else:
                sub_spaces.append(sub_space)
        return sub_spaces

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # not support

    # if hasattr(gym_spaces, "Text") and isinstance(gym_space, gym_spaces.Text):
    #    pass  # not support

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # not support

    # ---- other space
    try:
        flat_space = flatten_space(gym_space)
        if isinstance(flat_space, gym_spaces.Box):
            return srl_spaces.BoxSpace(flat_space.shape, flat_space.low, flat_space.high, flat_space.dtype)
    except NotImplementedError as e:
        logger.warning(f"Ignored for unsupported space. type '{type(gym_space)}', err_msg '{e}'")

    return None


def space_change_from_gym_to_srl(gym_space: gym_spaces.Space) -> SpaceBase:
    # tupleかdictがあればarrayにして管理、そうじゃない場合はそのまま
    srl_space = _space_change_from_gym_to_srl_sub(gym_space)
    assert srl_space is not None
    if isinstance(srl_space, list):
        srl_space = srl_spaces.ArraySpace(srl_space)
    return srl_space


def _space_encode_from_gym_to_srl_sub(gym_space: gym_spaces.Space, x: Any):
    # xは生データの可能性もあるので、最低限gymが期待している型に変換
    if isinstance(gym_space, gym_spaces.Discrete):
        return int(x)
    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        return np.asarray(x, dtype=gym_space.dtype)
    if isinstance(gym_space, gym_spaces.MultiBinary):
        return np.asarray(x, dtype=gym_space.dtype)
    if isinstance(gym_space, gym_spaces.Box):
        return np.asarray(x, dtype=gym_space.dtype)
    if isinstance(gym_space, gym_spaces.Tuple):
        s = cast(Any, gym_space.spaces)
        arr = []
        for space, x_part in zip(s, x):
            _x = _space_encode_from_gym_to_srl_sub(space, x_part)
            if _x is None:
                continue
            if isinstance(_x, list):
                arr.extend(_x)
            else:
                arr.append(_x)
        return arr
    if isinstance(gym_space, gym_spaces.Dict):
        arr = []
        for key in sorted(gym_space.spaces.keys()):
            _x = _space_encode_from_gym_to_srl_sub(gym_space.spaces[key], x[key])
            if _x is None:
                continue
            if isinstance(_x, list):
                arr.extend(_x)
            else:
                arr.append(_x)
        return arr

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # not support

    # if isinstance(gym_space, gym_spaces.Text):
    #    pass  # not support

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # not support

    # ---- other space
    try:
        x = flatten(gym_space, x)
        if isinstance(x, np.ndarray):
            return x
    except NotImplementedError as e:
        logger.debug(f"Ignored for unsupported space. type '{type(gym_space)}', '{x}', err_msg '{e}'")

    return None


def space_encode_from_gym_to_srl(gym_space: gym_spaces.Space, val: Any):
    x = _space_encode_from_gym_to_srl_sub(gym_space, val)
    assert x is not None, "Space flatten encode failed."
    return x


def _space_decode_to_srl_from_gym_sub(gym_space: gym_spaces.Space, x: Any, idx=0):
    if isinstance(gym_space, gym_spaces.Discrete):
        return x[idx], idx + 1
    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        return x[idx], idx + 1
    if isinstance(gym_space, gym_spaces.MultiBinary):
        return x[idx], idx + 1
    if isinstance(gym_space, gym_spaces.Box):
        return x[idx], idx + 1
    if isinstance(gym_space, gym_spaces.Tuple):
        arr = []
        for space in gym_space.spaces:
            y, idx = _space_decode_to_srl_from_gym_sub(space, x, idx)
            arr.append(y)
        return tuple(arr), idx

    if isinstance(gym_space, gym_spaces.Dict):
        keys = sorted(gym_space.spaces.keys())
        dic = {}
        for key in keys:
            y, idx = _space_decode_to_srl_from_gym_sub(gym_space.spaces[key], x, idx)
            dic[key] = y
        return dic, idx

    # if isinstance(gym_space, gym_spaces.Graph):
    #    pass  # not support

    # if isinstance(gym_space, gym_spaces.Text):
    #    pass  # TODO

    # if isinstance(gym_space, gym_spaces.Sequence):
    #    pass  # not support

    # 不明なのはsampleがあればそれを適用、なければNone
    if hasattr(gym_space, "sample"):
        y = gym_space.sample()
    else:
        y = None
    return y, idx


def space_decode_to_srl_from_gym(gym_space: gym_spaces.Space, srl_space: SpaceBase, val: Any) -> Any:
    if not isinstance(srl_space, srl_spaces.ArraySpace):
        val = [val]
    val, _ = _space_decode_to_srl_from_gym_sub(gym_space, val)
    assert val is not None, "Space flatten decode failed."
    return val


class GymWrapper(EnvBase):
    def __init__(self, config: EnvConfig):
        self.config = config

        self.seed = None
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
        logger.info(f"gym action_space: {self.env.action_space}")
        logger.info(f"gym obs_space   : {self.env.observation_space}")

        # metadata
        self.fps = 60
        self.render_mode = RenderModes.none
        self.render_modes = ["ansi", "human", "rgb_array"]
        if hasattr(self.env, "metadata"):
            logger.info(f"gym metadata    : {self.env.metadata}")
            self.fps = self.env.metadata.get("render_fps", 60)
            self.render_modes = self.env.metadata.get("render_modes", ["ansi", "human", "rgb_array"])

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
                        else:
                            _obs_type = EnvObservationTypes.IMAGE

                    if _obs_type != EnvObservationTypes.UNKNOWN:
                        # 画像はそのままのshape
                        self.enable_flatten_observation = False
                        _obs_space = srl_spaces.BoxSpace(
                            self.env.observation_space.shape,
                            self.env.observation_space.low,
                            self.env.observation_space.high,
                        )

            # --- space obs
            if _obs_type == EnvObservationTypes.UNKNOWN:
                self.enable_flatten_observation = True
                _obs_space = space_change_from_gym_to_srl(self.env.observation_space)
                if _obs_space.rl_type != RLTypes.DISCRETE:
                    if self.config.gym_prediction_by_simulation and self._pred_space_discrete():
                        _obs_type = EnvObservationTypes.DISCRETE
                    else:
                        _obs_type = EnvObservationTypes.CONTINUOUS
                else:
                    _obs_type = EnvObservationTypes.DISCRETE

        # --- space action
        if _act_space is None:
            self.enable_flatten_action = True
            _act_space = space_change_from_gym_to_srl(self.env.action_space)

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
            state = space_encode_from_gym_to_srl(self.env.observation_space, state)

        state = self.observation_space.sanitize(state)
        return state, info

    def step(self, action: EnvActionType) -> Tuple[np.ndarray, List[float], Union[bool, DoneTypes], InfoType]:
        # wrapper
        for w in self.config.gym_wrappers:
            action = w.action(action, self.env)

        # flatten
        if self.enable_flatten_action:
            action = space_decode_to_srl_from_gym(self.env.action_space, self.action_space, action)

        # step
        _t = self.env.step(action)
        if len(_t) == 4:
            state, reward, done, info = _t  # type: ignore
        else:
            state, reward, terminated, truncated, info = _t
            if terminated:
                done = DoneTypes.TERMINATED
            elif truncated:
                done = DoneTypes.TRUNCATED
            else:
                done = DoneTypes.NONE

        # wrapper
        for w in self.config.gym_wrappers:
            state = w.observation(state, self.env)
            reward = w.reward(reward, self.env)
            done = w.done(cast(DoneTypes, done), self.env)

        # flatten
        if self.enable_flatten_observation:
            state = space_encode_from_gym_to_srl(self.env.observation_space, state)

        state = self.observation_space.sanitize(state)
        return state, [float(reward)], done, info

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def close(self) -> None:
        # render 内で使われている pygame に対して close -> init をするとエラーになる
        # Fatal Python error: (pygame parachute) Segmentation Fault
        self.env.close()
        pass

    @property
    def unwrapped(self) -> object:
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
