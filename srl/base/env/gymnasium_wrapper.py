import logging
import os
import pickle
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import gymnasium
import numpy as np
from gymnasium import spaces as gym_spaces
from gymnasium.spaces import flatten, flatten_space

from srl.base import spaces as srl_spaces
from srl.base.define import EnvActionType, KeyBindType, RenderModes, SpaceTypes
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.config import EnvConfig

if TYPE_CHECKING:
    from srl.base.rl.worker import RLWorker


logger = logging.getLogger(__name__)


"""
・gym_spaceを1次元にして管理する
・decodeもあるので順序を保持する
・変換できないものはエラーログを出力して無視する
"""


def _space_change_from_gym_to_srl_sub(gym_space: gym_spaces.Space) -> Optional[Union[SpaceBase, List[SpaceBase]]]:
    if isinstance(gym_space, gym_spaces.Discrete):
        if hasattr(gym_space, "start"):
            return srl_spaces.DiscreteSpace(int(gym_space.n), start=int(gym_space.start))
        else:
            return srl_spaces.DiscreteSpace(int(gym_space.n))

    if isinstance(gym_space, gym_spaces.MultiDiscrete):
        return srl_spaces.BoxSpace(gym_space.shape, 0, gym_space.nvec, np.int64, SpaceTypes.DISCRETE)

    if isinstance(gym_space, gym_spaces.MultiBinary):
        return srl_spaces.BoxSpace(gym_space.shape, 0, 1, np.int8, SpaceTypes.DISCRETE)

    if isinstance(gym_space, gym_spaces.Box):
        # image check
        _obs_type = SpaceTypes.UNKNOWN
        if "uint" in str(gym_space.dtype):
            if len(gym_space.shape) == 2:
                _obs_type = SpaceTypes.GRAY_2ch
            elif len(gym_space.shape) == 3:
                # w,h,ch 想定
                ch = gym_space.shape[-1]
                if ch == 1:
                    _obs_type = SpaceTypes.GRAY_3ch
                elif ch == 3:
                    _obs_type = SpaceTypes.COLOR
                else:
                    _obs_type = SpaceTypes.IMAGE
        return srl_spaces.BoxSpace(gym_space.shape, gym_space.low, gym_space.high, gym_space.dtype, _obs_type)

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

    if isinstance(gym_space, gym_spaces.Graph):
        pass  # not support

    if isinstance(gym_space, gym_spaces.Text):
        shape = (gym_space.max_length,)
        # TODO TextSpace
        return srl_spaces.BoxSpace(shape, 0, len(gym_space.character_set), np.int64, SpaceTypes.DISCRETE)

    if isinstance(gym_space, gym_spaces.Sequence):
        pass  # not support

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
        srl_space = srl_spaces.MultiSpace(srl_space)
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

    if isinstance(gym_space, gym_spaces.Graph):
        pass  # not support

    if isinstance(gym_space, gym_spaces.Text):
        arr = [gym_space.character_index(v) for v in x]
        return arr

    if isinstance(gym_space, gym_spaces.Sequence):
        pass  # not support

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

    if isinstance(gym_space, gym_spaces.Graph):
        pass  # not support

    if isinstance(gym_space, gym_spaces.Text):
        pass  # TODO

    if isinstance(gym_space, gym_spaces.Sequence):
        pass  # not support

    # 不明なのはsampleがあればそれを適用、なければNone
    if hasattr(gym_space, "sample"):
        y = gym_space.sample()
    else:
        y = None
    return y, idx


def space_decode_to_srl_from_gym(gym_space: gym_spaces.Space, srl_space: SpaceBase, val: Any) -> Any:
    if not isinstance(srl_space, srl_spaces.MultiSpace):
        val = [val]
    val, _ = _space_decode_to_srl_from_gym_sub(gym_space, val)
    assert val is not None, "Space flatten decode failed."
    return val


class GymnasiumWrapper(EnvBase):
    is_print_log = True

    def __init__(self, config: EnvConfig):
        self.config = config

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        if GymnasiumWrapper.is_print_log:
            logger.info("set SDL_VIDEODRIVER='dummy'")

        self.env = self.make_gymnasium_env()
        if GymnasiumWrapper.is_print_log:
            logger.info(f"gym action_space: {self.env.action_space}")
            logger.info(f"gym obs_space   : {self.env.observation_space}")

        # metadata
        self.fps = 60
        self.render_mode = RenderModes.none
        self.render_modes = ["ansi", "human", "rgb_array"]
        if hasattr(self.env, "metadata"):
            if GymnasiumWrapper.is_print_log:
                logger.info(f"gym metadata    : {self.env.metadata}")
            self.fps = self.env.metadata.get("render_fps", 60)
            self.render_modes = self.env.metadata.get("render_modes", ["ansi", "human", "rgb_array"])

        # --- wrapper
        act_space = None
        obs_space = None
        self.use_wrapper_act = False
        self.use_wrapper_obs = False
        if config.gym_wrapper is not None:
            act_space = config.gym_wrapper.remap_action_space(self.env)
            obs_space = config.gym_wrapper.remap_observation_space(self.env)
            self.use_wrapper_act = act_space is not None
            self.use_wrapper_obs = obs_space is not None

        # --- space
        if act_space is None:
            act_space = space_change_from_gym_to_srl(self.env.action_space)
        if obs_space is None:
            obs_space = space_change_from_gym_to_srl(self.env.observation_space)

        self._action_space: SpaceBase = act_space
        self._observation_space: SpaceBase = obs_space

        if GymnasiumWrapper.is_print_log:
            logger.info(f"use wrapper act: {self.use_wrapper_act}")
            logger.info(f"action         : {self._action_space}")
            logger.info(f"use wrapper obs: {self.use_wrapper_obs}")
            logger.info(f"observation    : {self._observation_space}")
        GymnasiumWrapper.is_print_log = False

    def make_gymnasium_env(self, **kwargs) -> gymnasium.Env:
        if self.config.gym_make_func is None:
            return gymnasium.make(self.config.name, **self.config.kwargs, **kwargs)
        env = self.config.gym_make_func(self.config.name, **self.config.kwargs, **kwargs)
        return cast(gymnasium.Env, env)

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
    def max_episode_steps(self) -> int:
        if hasattr(self.env, "_max_episode_steps"):
            return getattr(self.env, "_max_episode_steps")
        elif hasattr(self.env, "spec"):
            if self.env.spec is not None and self.env.spec.max_episode_steps is not None:
                return self.env.spec.max_episode_steps
        return 99_999

    @property
    def player_num(self) -> int:
        return 1

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        if seed is None:
            state, info = self.env.reset()
        else:
            state, info = self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

        if self.use_wrapper_obs:
            assert self.config.gym_wrapper is not None
            state = self.config.gym_wrapper.remap_observation(state, self.env)
        else:
            state = space_encode_from_gym_to_srl(self.env.observation_space, state)

        state = self.observation_space.sanitize(state)
        self.info.set_dict(info)
        return state

    def step(self, action: EnvActionType) -> Tuple[Any, List[float], bool, bool]:
        if self.use_wrapper_act:
            assert self.config.gym_wrapper is not None
            action = self.config.gym_wrapper.remap_action(action, self.env)
        else:
            action = space_decode_to_srl_from_gym(self.env.action_space, self.action_space, action)

        # step
        state, reward, terminated, truncated, info = self.env.step(action)

        if self.use_wrapper_obs:
            assert self.config.gym_wrapper is not None
            state = self.config.gym_wrapper.remap_observation(state, self.env)
        else:
            state = space_encode_from_gym_to_srl(self.env.observation_space, state)

        if self.config.gym_wrapper is not None:
            reward = self.config.gym_wrapper.remap_reward(cast(float, reward), self.env)
            terminated, truncated = self.config.gym_wrapper.remap_done(terminated, truncated, self.env)

        state = self.observation_space.sanitize(state)
        self.info.set_dict(info)
        return state, [float(reward)], terminated, truncated

    def close(self) -> None:
        try:
            self.env.close()
        except Exception as e:
            logger.error(e)

    @property
    def unwrapped(self) -> object:
        return self.env

    def setup(self, **kwargs):
        render_mode = kwargs.get("render_mode", RenderModes.none)
        render_mode = RenderModes.from_str(render_mode)
        # --- terminal
        # modeが違っていたら作り直す
        if (
            (render_mode in [RenderModes.terminal])
            and (self.render_mode != RenderModes.terminal)
            and ("ansi" in self.render_modes)
        ):
            try:
                self.env.close()
            except Exception as e:
                logger.warning(e)
            self.env = self.make_gymnasium_env(render_mode="ansi")
            self.render_mode = RenderModes.terminal

        # --- rgb_array
        # modeが違っていたら作り直す
        if (
            (render_mode in [RenderModes.rgb_array, RenderModes.window])
            and (self.render_mode != RenderModes.rgb_array)
            and ("rgb_array" in self.render_modes)
        ):
            try:
                self.env.close()
            except Exception as e:
                logger.warning(e)
            self.env = self.make_gymnasium_env(render_mode="rgb_array")
            self.render_mode = RenderModes.rgb_array

        # --- unwrapped function
        if hasattr(self.env.unwrapped, "setup"):
            return self.env.unwrapped.setup(**kwargs)  # type: ignore

    def backup(self) -> Any:
        if hasattr(self.env.unwrapped, "backup"):
            return self.env.unwrapped.backup()  # type: ignore
        else:
            return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        if hasattr(self.env.unwrapped, "restore"):
            return self.env.unwrapped.restore(data)  # type: ignore
        else:
            self.env: gymnasium.Env = pickle.loads(data)

    def get_invalid_actions(self, player_index: int = -1) -> List[EnvActionType]:
        if hasattr(self.env.unwrapped, "get_invalid_actions"):
            return self.env.unwrapped.get_invalid_actions()  # type: ignore
        else:
            return []

    def action_to_str(self, action: Union[str, EnvActionType]) -> str:
        if hasattr(self.env.unwrapped, "action_to_str"):
            return self.env.unwrapped.action_to_str(action)  # type: ignore
        else:
            return str(action)

    def get_key_bind(self) -> Optional[KeyBindType]:
        if hasattr(self.env.unwrapped, "get_key_bind"):
            return self.env.unwrapped.get_key_bind()  # type: ignore
        else:
            return None

    def make_worker(self, name: str, **kwargs) -> Optional["RLWorker"]:
        if hasattr(self.env.unwrapped, "make_worker"):
            return self.env.unwrapped.make_worker(name, **kwargs)  # type: ignore
        else:
            return None

    @property
    def render_interval(self) -> float:
        return 1000 / self.fps

    def render_terminal(self, **kwargs) -> None:
        if self.render_mode == RenderModes.terminal:
            print(self.env.render(**kwargs))

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if self.render_mode == RenderModes.rgb_array:
            return np.asarray(self.env.render(**kwargs))
        return None
