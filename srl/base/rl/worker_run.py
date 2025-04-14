import logging
from typing import Any, Dict, Generic, List, Optional, Union, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import DoneTypes, EnvActionType, RenderModes
from srl.base.env.env_run import EnvRun
from srl.base.exception import SRLError
from srl.base.info import Info
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import DummyRLWorker, RLWorkerGeneric
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase, TActSpace, TActType, TObsSpace, TObsType
from srl.utils import render_functions as render_funcs

logger = logging.getLogger(__name__)


class WorkerRun(Generic[TActSpace, TActType, TObsSpace, TObsType]):
    def __init__(
        self,
        worker: RLWorkerGeneric[RLConfig, RLParameter, RLMemory, TActSpace, TActType, TObsSpace, TObsType],
        env: EnvRun,
    ):
        # - restore/backup用に状態は意識して管理
        # - env関係の値の保存はprocessorで変更がありそうなものはこちらが持つ
        #   そうじゃないものはenvの値をそのまま利用
        # - baseの中はクラスの中身まで把握しているものとして速度優先でprivateな変数のアクセスをする
        #   例えば self.env.info と self._env.env.info は同じだけど基本後半を使う方針

        worker.config.setup(env, enable_log=False)
        worker._set_worker_run(self)

        self._worker = worker
        self._config: RLConfig[SpaceBase, SpaceBase] = worker.config
        self._env = env
        self._render = Render(worker)
        self._is_setup = False

        self._setup_val(RunContext())
        self._reset_val(0)

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def worker(self):
        return self._worker

    @property
    def config(self) -> RLConfig:
        return self._config

    @property
    def env(self) -> EnvRun:
        return self._env

    @property
    def context(self) -> RunContext:
        return self._context

    @property
    def distributed(self) -> bool:
        return self._context.distributed

    @property
    def training(self) -> bool:
        return self._context.training

    @property
    def train_only(self) -> bool:
        return self._context.train_only

    @property
    def rollout(self) -> bool:
        return self._context.rollout

    @property
    def rendering(self) -> bool:
        return self._context.rendering

    @property
    def actor_id(self) -> int:
        return self._context.actor_id

    @property
    def player_index(self) -> int:
        return self._player_index

    @property
    def info(self) -> Info:
        return self._worker.info

    @property
    def prev_state(self) -> TObsType:
        return self._prev_state

    @property
    def state(self) -> TObsType:
        return self._state

    @property
    def state_one_step(self) -> TObsType:
        return self._one_states[-1] if self._use_stacked_state else self._state

    @property
    def prev_render_image_state(self) -> np.ndarray:
        return self._prev_render_image

    @property
    def render_image_state(self) -> np.ndarray:
        return self._render_image

    @property
    def render_image_state_one_step(self) -> np.ndarray:
        return self._one_render_images[-1] if self._use_stacked_render_image else self._render_image

    @property
    def prev_action(self) -> TActType:
        return self._prev_action

    def get_onehot_prev_action(self):
        return self._config.action_space.get_onehot(self._prev_action)

    @property
    def action(self) -> TActType:
        return self._action

    def get_onehot_action(self, action=None):
        return self._config.action_space.get_onehot(self._action if action is None else action)

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._env._done != DoneTypes.NONE

    @property
    def terminated(self) -> bool:
        return self._env._done == DoneTypes.TERMINATED

    @property
    def done_type(self) -> DoneTypes:
        return self._env._done

    @property
    def done_reason(self) -> str:
        return self._env.env.done_reason

    @property
    def prev_invalid_actions(self) -> List[TActType]:
        return self._prev_invalid_actions

    @property
    def invalid_actions(self) -> List[TActType]:
        return self._invalid_actions

    @property
    def total_step(self) -> int:
        return self._total_step

    @property
    def episode_seed(self) -> Optional[int]:
        return self._episode_seed

    def setup(
        self,
        context: Optional[RunContext] = None,
        render_mode: Union[str, RenderModes] = RenderModes.none,
    ):
        if context is None:
            context = RunContext(self.env.config, self._config)
        if render_mode == RenderModes.none:
            render_mode = context.render_mode

        self._setup_val(context)
        self._render.set_render_mode(render_mode, enable_window=False)
        logger.debug(f"on_setup: {render_mode=}")
        self._worker.on_setup(self, context)
        self._is_setup = True

    def _setup_val(self, context: RunContext):
        self._context = context
        self._total_step: int = 0

        self._use_stacked_state = self._config.window_length > 1
        self._use_render_image = self._config.use_render_image_state()
        self._use_stacked_render_image = self._use_render_image and (self._config.render_image_window_length > 1)

        self._tracking_episode = False
        self._tracking_size = 0  # 0: no tracking, -1: episode_tracking

    def teardown(self):
        logger.debug("teardown")
        self._worker.on_teardown(self)
        self._is_setup = False

    def reset(self, player_index: int, seed: Optional[int] = None) -> None:
        logger.debug(f"worker_run.on_reset: {player_index=}, {seed=}")
        if not self._is_setup:
            raise SRLError("Cannot call worker.on_reset() before calling worker.setup()")

        if self._context.rendering:
            self._render.cache_reset()

        self._reset_val(player_index, seed)

    def _reset_val(self, player_index: int, seed: Optional[int] = None):
        self._player_index = player_index
        self._episode_seed = seed
        self._is_reset = False
        self._is_ready_policy = False

        # --- state
        self._state = self._config.observation_space.get_default()
        self._prev_state = self._config.observation_space.get_default()
        self._one_states = [self._config.observation_space_one_step.get_default() for _ in range(self._config.window_length)]

        if self._use_render_image:
            self._render_image = self._config.obs_render_img_space.get_default()
            self._prev_render_image = self._config.obs_render_img_space.get_default()
            self._one_render_images = [self._config.obs_render_img_space_one_step.get_default() for _ in range(self._config.render_image_window_length)]
        else:
            self._render_image = np.zeros((1,))
            self._prev_render_image = np.zeros((1,))
            self._one_render_images = []

        # action, reward, done
        self._action = self._config.action_space.get_default()
        self._prev_action = self._config.action_space.get_default()
        self._step_reward: float = 0.0
        self._reward: float = 0.0
        self._invalid_actions: List[TActType] = []
        self._prev_invalid_actions: List[TActType] = []

        # tracking
        if self._tracking_episode:
            self._tracking_state: List[TObsType] = []
            self._tracking_render_image: List[np.ndarray] = []
            self._tracking_action: List[TActType] = []
            self._tracking_reward: List[float] = []
            self._tracking_terminate: List[int] = []
            self._tracking_invalid_actions: List[List[TActType]] = []
            self._tracking_user_data: List[Dict[str, Any]] = []

    def ready_policy(self):
        """policyの準備を実行。1週目は on_reset、2週目以降は on_step を実行。"""
        # encode -> set invalid -> on_step -> step_reward=0
        if self._is_ready_policy:
            raise SRLError("'ready_policy' cannot be called consecutively.")

        # --- state
        self._prev_state = self._state
        state = cast(TObsType, self._config.state_encode_one_step(self.env.state, self._env))
        if self._use_stacked_state:
            del self._one_states[0]
            self._one_states.append(state)
            self._state = self._config.observation_space_one_step.encode_stack(self._one_states)
        else:
            self._state = state

        # --- render image
        if self._use_render_image:
            self._prev_render_image = self._render_image
            render_image = self._config.render_image_state_encode_one_step(self._env)
            if self._use_stacked_render_image:
                del self._one_render_images[0]
                self._one_render_images.append(render_image)
                self._render_image = self._config.obs_render_img_space_one_step.encode_stack(self._one_render_images)
            else:
                self._render_image = render_image

        # --- invalid_actions
        self._prev_invalid_actions = self._invalid_actions
        self._invalid_actions = [
            cast(TActType, self._config.action_encode(a))
            for a in self._env.get_invalid_actions(self._player_index)  #
        ]

        # --- tracking
        if self._tracking_episode:
            # state (+1)
            if (self._tracking_size != -1) and (len(self._tracking_state) == self._tracking_size + 1):
                del self._tracking_state[0]
            self._tracking_state.append(self._state)
            # render image
            if self._use_render_image:
                if (self._tracking_size != -1) and (len(self._tracking_render_image) == self._tracking_size + 1):
                    del self._tracking_render_image[0]
                self._tracking_render_image.append(self._render_image)

        # --- reset/step
        if not self._is_reset:
            logger.debug("on_reset")
            self._worker.on_reset(self)
            self._is_reset = True
        else:
            self._reward = self._step_reward
            self._step_reward = 0.0

            if self._tracking_episode:
                if (self._tracking_size != -1) and (len(self._tracking_reward) == self._tracking_size):
                    del self._tracking_reward[0]
                    del self._tracking_terminate[0]
                    del self._tracking_invalid_actions[0]
                self._tracking_invalid_actions.append(self._invalid_actions[:])
                self._tracking_reward.append(self._reward)
                self._tracking_terminate.append(int(self.terminated))

            logger.debug("on_step")
            self._worker.on_step(self)

        self._is_ready_policy = True

    def policy(self, call_ready_policy: bool = True) -> EnvActionType:
        if call_ready_policy:
            self.ready_policy()
        if not self._is_ready_policy:
            raise SRLError("Please call 'worker.ready_policy' first.")

        logger.debug("policy")
        action = self._worker.policy(self)
        self._prev_action = self._action
        self._action = action

        if self._tracking_episode:
            if len(self._tracking_action) == self._tracking_size:
                del self._tracking_action[0]
            self._tracking_action.append(action)

        env_action = self._config.action_decode(action)

        if self._context.rendering:
            self._render.cache_reset()

        self._is_ready_policy = False
        return env_action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return
        self._total_step += 1

        # 相手の番のrewardも加算
        self._step_reward += self._env.rewards[self.player_index]

        if self._env._done != DoneTypes.NONE:
            # 終了後の状態を取得
            self.ready_policy()
            if self._context.rendering:
                self._render.cache_reset()

    # ------------------------------------
    # invalid
    # ------------------------------------
    def get_invalid_actions(self, env: Optional[EnvRun] = None, player_index: int = -1) -> List[TActType]:
        if env is None:
            return self._invalid_actions
        return [
            cast(TActType, self._config.action_encode(a))
            for a in env.get_invalid_actions(player_index)  #
        ]

    def get_valid_actions(self, env: Optional[EnvRun] = None, player_index: int = -1) -> List[TActType]:
        return self._config.action_space.get_valid_actions(self.get_invalid_actions(env, player_index))

    def add_invalid_actions(self, invalid_actions: List[TActType], encode: bool = False) -> None:
        if encode:
            invalid_actions = [self._config.action_encode(a) for a in invalid_actions]  # type: ignore
        self._invalid_actions = list(set(self._invalid_actions + invalid_actions))

    # ------------------------------------
    # render functions
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        self._render.set_render_options(interval, scale, font_name, font_size)

    def render(self, **kwargs):
        logger.debug("render")
        return self._render.render(worker=self, **kwargs)

    def render_terminal_text(self, **kwargs) -> str:
        if not self._is_reset:  # on_reset前はrenderしない
            return ""
        return self._render.render_terminal_text(worker=self, **kwargs)

    def render_terminal_text_to_image(self, **kwargs) -> Optional[np.ndarray]:
        if not self._is_reset:  # on_reset前はrenderしない
            return None
        return self._render.render_terminal_text_to_image(worker=self, **kwargs)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if not self._is_reset:  # on_reset前はrenderしない
            return None
        return self._render.render_rgb_array(worker=self, **kwargs)

    def render_rl_image(self) -> Optional[np.ndarray]:
        if not self._config.observation_space_one_step.is_image():
            return None
        space = cast(BoxSpace, self._config.observation_space_one_step)
        if self._use_stacked_state:
            img = cast(np.ndarray, self._one_states[-1]).copy()
        else:
            img = cast(np.ndarray, self._state).copy()
        return space.to_image(img)

    def create_render_image(
        self,
        add_terminal: bool = True,
        add_rgb_array: bool = True,
        info_text: str = "",
    ) -> np.ndarray:
        """
        ---------------------------------
        | BG white   | BG black         |
        | [env]      | [info]           |
        | [rl state] | [rl render text] |
        |            | [rl render rgb]  |
        ---------------------------------
        """
        padding = 1
        border_color = (111, 175, 0)

        # --- env image
        env_img = self._env.render_rgb_array()
        if env_img is None:
            env_img = self._env.render_terminal_text_to_image()
        assert env_img is not None
        env_img = render_funcs.add_padding(env_img, padding, padding, padding, padding, border_color)

        # [rl state]
        if self._config.render_rl_image_size:
            # 同じ場合は省略
            if self.env.observation_space != self.config._rl_obs_space_one_step:
                rl_state_img = self.render_rl_image()
                if rl_state_img is not None:
                    import cv2

                    rl_state_img = cv2.resize(rl_state_img, self._config.render_rl_image_size, interpolation=cv2.INTER_LINEAR)
                    rl_state_img = render_funcs.add_padding(rl_state_img, padding, padding, padding, padding, border_color)
                    env_img = render_funcs.vconcat(env_img, rl_state_img)

        # [info]
        rl_img = None
        if info_text != "":
            info_img = render_funcs.text_to_rgb_array(info_text, self._render.font_name, self._render.font_size)
            info_img = render_funcs.add_padding(info_img, padding, padding, padding, padding)
            rl_img = info_img

        # [rl render text]
        if add_terminal:
            t_img = self.render_terminal_text_to_image()
            if t_img is not None:
                t_img = render_funcs.add_padding(t_img, padding, padding, padding, padding)
                if rl_img is None:
                    rl_img = t_img
                else:
                    rl_img = render_funcs.vconcat(rl_img, t_img)

        # [rl render rgb]
        if add_rgb_array:
            rl_render = self.render_rgb_array()
            if rl_render is not None:
                rl_render = render_funcs.add_padding(rl_render, padding, padding, padding, padding)
                if rl_img is None:
                    rl_img = rl_render
                else:
                    rl_img = render_funcs.vconcat(rl_img, rl_render)

        # --- env + rl
        if rl_img is not None:
            env_img = render_funcs.hconcat(env_img, rl_img)

        return env_img

    # ------------------------------------
    # tracking
    # ------------------------------------
    def enable_tracking(self, max_size: int = -1):
        """stateだけ max_size+1 確保"""
        if self._is_setup:
            raise SRLError("Please call it in 'on_setup()'")
        self._tracking_episode = True
        self._tracking_size = max_size

    def update_tracking_data(self, data: Dict[str, Any], index: int = -1):
        self._tracking_user_data[index].update(data)

    def add_tracking_data(self, data: Dict[str, Any]):
        if len(self._tracking_user_data) == self._tracking_size:
            del self._tracking_user_data[0]
        self._tracking_user_data.append(data)

    def get_tracking(self, key: str, size: int = -1, dummy: Any = None) -> List[Any]:
        if key == "state":
            arr = self._tracking_state
            if dummy is None:
                dummy = self._config.observation_space.get_default()
        elif key == "render_image":
            arr = self._tracking_render_image
            if dummy is None:
                dummy = self._config.obs_render_img_space.get_default()
        elif key == "action":
            arr = self._tracking_action
            if dummy is None:
                dummy = self._config.action_space.get_default()
        elif key == "invalid_actions":
            arr = self._tracking_invalid_actions
            if dummy is None:
                dummy = []
        elif key == "reward":
            arr = self._tracking_reward
            if dummy is None:
                dummy = 0.0
        elif key == "terminated":
            arr = self._tracking_terminate
            if dummy is None:
                dummy = 0
        else:
            raise ValueError(key)

        if size > 0:
            if len(arr) < size:
                return [dummy] * (size - len(arr)) + arr
            else:
                return arr[-size:]
        else:
            return arr[:]

    def get_tracking_data(self, key: str, size: int = 0, dummy: Any = None) -> List[Any]:
        if size > 0:
            if len(self._tracking_user_data) < size:
                arr = [d[key] if key in d else dummy for d in self._tracking_user_data[:]]
                return [dummy for _ in range(size - len(arr))] + arr[:]
            else:
                return [d[key] if key in d else dummy for d in self._tracking_user_data[-size:]]
        return [d[key] if key in d else dummy for d in self._tracking_user_data]

    def add_tracking_dummy_step(self, state=None, action=None, invalid_actions=[], reward: float = 0.0, terminated: bool = False, tracking_data: Dict[str, Any] = {}, is_reset: bool = False):
        if state is None:
            state = self._config.observation_space.get_default()
        if (self._tracking_size != -1) and (len(self._tracking_state) == self._tracking_size + 1):
            del self._tracking_state[0]
        # reset時は初期stateの前に追加
        if is_reset:
            self._tracking_state.insert(-1, state)
        else:
            self._tracking_state.append(state)
        if self._use_render_image:
            render_image = self._config.obs_render_img_space.get_default()
            if (self._tracking_size != -1) and (len(self._tracking_render_image) == self._tracking_size + 1):
                del self._tracking_render_image[0]
            if is_reset:
                self._tracking_render_image.insert(-1, render_image)
            else:
                self._tracking_render_image.append(render_image)

        if action is None:
            action = self._config.action_space.get_default()
        if (self._tracking_size != -1) and (len(self._tracking_action) == self._tracking_size):
            del self._tracking_action[0]
            del self._tracking_invalid_actions[0]
            del self._tracking_reward[0]
            del self._tracking_terminate[0]
            del self._tracking_user_data[0]
        self._tracking_action.append(action)
        self._tracking_invalid_actions.append(invalid_actions)
        self._tracking_reward.append(reward)
        self._tracking_terminate.append(int(terminated))
        self._tracking_user_data.append(tracking_data)

    # ------------------------------------
    # backup/restore
    # ------------------------------------
    def backup(self) -> Any:
        logger.debug(f"backup: step={self._total_step}")
        d = [
            # setup
            self._is_setup,
            self._total_step,
            self._use_stacked_state,
            self._use_render_image,
            self._use_stacked_render_image,
            self._tracking_episode,
            self._tracking_size,
            # reset
            self._player_index,
            self._episode_seed,
            self._is_reset,
            self._is_ready_policy,
            self._config.observation_space.copy_value(self._state),
            self._config.observation_space.copy_value(self._prev_state),
            [self._config.observation_space_one_step.copy_value(s) for s in self._one_states],
            self._config.obs_render_img_space.copy_value(self._render_image),
            self._config.obs_render_img_space.copy_value(self._prev_render_image),
            [self._config.obs_render_img_space_one_step.copy_value(s) for s in self._one_render_images],
            self._config.action_space.copy_value(self._action),
            self._config.action_space.copy_value(self._prev_action),
            self._step_reward,
            self._reward,
            self._invalid_actions[:],
            self._prev_invalid_actions[:],
            # env
            self._env.backup(),
        ]
        if self._tracking_episode:
            d += [
                [self._config.observation_space.copy_value(s) for s in self._tracking_state],
                [self._config.obs_render_img_space.copy_value(s) for s in self._tracking_render_image],
                [self._config.action_space.copy_value(d) for d in self._tracking_action],
                self._tracking_reward[:],
                self._tracking_terminate[:],
                self._tracking_invalid_actions[:],
                [d.copy() for d in self._tracking_user_data],
            ]

        return d

    def restore(self, dat: Any):
        logger.debug(f"restore: step={dat[0]}")
        # setup
        self._is_setup = dat[0]
        self._total_step = dat[1]
        self._use_stacked_state = dat[2]
        self._use_render_image = dat[3]
        self._use_stacked_render_image = dat[4]
        self._tracking_episode = dat[5]
        self._tracking_size = dat[6]
        # reset
        self._player_index = dat[7]
        self._episode_seed = dat[8]
        self._is_reset = dat[9]
        self._is_ready_policy = dat[10]
        self._state = dat[11]
        self._prev_state = dat[12]
        self._one_states = dat[13]
        self._render_image = dat[14]
        self._prev_render_image = dat[15]
        self._one_render_images = dat[16]
        self._action = dat[17]
        self._prev_action = dat[18]
        self._step_reward = dat[19]
        self._reward = dat[20]
        self._invalid_actions = dat[21]
        self._prev_invalid_actions = dat[22]
        # env
        self._env.restore(dat[23])
        # tracking
        if self._tracking_episode:
            self._tracking_state = dat[24][:]
            self._tracking_render_image = dat[25][:]
            self._tracking_action = dat[26][:]
            self._tracking_reward = dat[27][:]
            self._tracking_terminate = dat[28][:]
            self._tracking_invalid_actions = dat[29][:]
            self._tracking_user_data = dat[30][:]

        if self._context.rendering:
            self._render.cache_reset()

    def copy(self) -> "WorkerRun":
        worker = WorkerRun(DummyRLWorker(self._config), self.env)
        dat = self.backup()
        worker.restore(dat)
        return worker

    # ------------------------------------
    # utils
    # ------------------------------------
    def sample_action(self) -> TActType:
        action = cast(EnvActionType, self._config.action_space_of_env.sample(self.get_invalid_actions()))
        return cast(TActType, self._config.action_encode(action))

    def sample_action_for_env(self) -> EnvActionType:
        return self._env.sample_action()

    def override_action(self, env_action: EnvActionType, encode: bool = True) -> TActType:
        if encode:
            rl_action = cast(TActType, self._config.action_encode(env_action))
        else:
            rl_action = cast(TActType, env_action)
        self._tracking_action[-1] = rl_action
        return rl_action
