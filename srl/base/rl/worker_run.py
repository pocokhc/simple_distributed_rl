import logging
import traceback
from typing import Any, Dict, Generic, List, Literal, Optional, cast

import numpy as np

from srl.base.context import RunContext, RunState
from srl.base.define import DoneTypes, EnvActionType, RenderModeType, RLActionType
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
        # - env関係の値はenv側とタイミングがずれる場合があるのでこちらで別途保存して利用
        # - baseの中はクラスの中身まで把握しているものとして速度優先でprivateな変数のアクセスをする
        #   例えば self.env.info と self._env.env.info は同じだけど基本後半を使う方針

        worker.config.setup(env, enable_log=False)
        worker._set_worker_run(self)

        self._worker = worker
        self._config: RLConfig[SpaceBase, SpaceBase] = worker.config
        self._env = env
        self._render = Render(worker)
        self._is_setup = False

        self._setup_val(RunContext(), RunState())
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
        return self._context.rl_render_mode != ""

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
    def run_state(self) -> RunState:
        return self._run_state

    @property
    def train_count(self) -> int:
        return self._run_state.train_count

    @property
    def prev_state(self) -> TObsType:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return None  # type: ignore  # on_stepはNone
        else:
            return self._prev_state

    @property
    def state(self) -> TObsType:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._prev_state
        else:
            return self._state

    @property
    def next_state(self) -> TObsType:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._state
        else:
            return None  # type: ignore  # on_step以外はNone

    def get_state_one_step(self, idx: int = -1) -> TObsType:
        return self._one_states[idx] if self._use_stacked_state else self._state

    @property
    def prev_render_image_state(self) -> np.ndarray:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return None  # type: ignore  # on_stepはNone
        else:
            return self._prev_render_image

    @property
    def render_image_state(self) -> np.ndarray:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._prev_render_image
        else:
            return self._render_image

    @property
    def next_render_image_state(self) -> np.ndarray:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._render_image
        else:
            return None  # type: ignore  # on_step以外はNone

    def get_render_image_state_one_step(self, idx: int = -1) -> np.ndarray:
        return self._one_render_images[idx] if self._use_stacked_render_image else self._render_image

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
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return None  # type: ignore  # on_stepはNone
        else:
            return self._prev_invalid_actions

    @property
    def invalid_actions(self) -> List[TActType]:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._prev_invalid_actions
        else:
            return self._invalid_actions

    @property
    def next_invalid_actions(self) -> List[TActType]:
        # on_stepの中はずらす
        if self._on_step_in_progress:
            return self._invalid_actions
        else:
            return None  # type: ignore  # on_step以外はNone

    @property
    def step_in_training(self) -> int:
        return self._step_in_training

    @property
    def step_in_episode(self) -> int:
        return self._step_in_episode

    @property
    def episode_seed(self) -> Optional[int]:
        return self._episode_seed

    def setup(
        self,
        context: Optional[RunContext] = None,
        render_mode: RenderModeType = "",
        run_state: Optional[RunState] = None,
    ):
        if context is None:
            context = RunContext(self.env.config, self._config)
        if render_mode == "":
            render_mode = context.rl_render_mode
        if render_mode == "window":  # rlはwindowは使わない
            render_mode = "rgb_array"
        if run_state is None:
            run_state = RunState()

        self._setup_val(context, run_state)
        self._render.set_render_mode(render_mode)
        logger.debug(f"on_setup: {render_mode=}")
        self._worker.on_setup(self, context)
        self._is_setup = True

    def _setup_val(self, context: RunContext, run_state: RunState):
        self._context = context
        self._run_state = run_state
        self._step_in_training: int = 0

        self._use_stacked_state = self._config.window_length > 1
        self._use_render_image = self._config.use_render_image_state()
        self._use_stacked_render_image = self._use_render_image and (self._config.render_image_window_length > 1)

        self._tracking_size = -1

    def teardown(self):
        logger.debug("teardown")
        self._worker.on_teardown(self)
        self._is_setup = False

    def reset(self, player_index: int, seed: Optional[int] = None) -> None:
        logger.debug(f"worker_run.on_reset: {player_index=}, {seed=}")
        if not self._is_setup:
            raise SRLError("Cannot call worker.on_reset() before calling worker.setup()")

        if self._render.rendering:
            self._render.cache_reset()

        self._reset_val(player_index, seed)

    def _reset_val(self, player_index: int, seed: Optional[int] = None):
        self._player_index = player_index
        self._episode_seed = seed
        self._is_reset = False
        self._step_in_episode = 0
        self._on_step_in_progress = False  # backup不要

        # --- state
        self._prev_state: TObsType = self._config.observation_space.get_default()
        self._state: TObsType = self._config.observation_space.get_default()
        self._one_states = [self._config.observation_space_one_step.get_default() for _ in range(self._config.window_length)]

        if self._use_render_image:
            self._prev_render_image: np.ndarray = self._config.obs_render_img_space.get_default()
            self._render_image: np.ndarray = self._config.obs_render_img_space.get_default()
            self._one_render_images = [self._config.obs_render_img_space_one_step.get_default() for _ in range(self._config.render_image_window_length)]
        else:
            self._prev_render_image: np.ndarray = np.zeros((1,))
            self._render_image: np.ndarray = np.zeros((1,))
            self._one_render_images = []

        # action, reward, done
        self._prev_action: TActType = self._config.action_space.get_default()
        self._action: TActType = self._config.action_space.get_default()
        self._step_reward: float = 0.0
        self._reward: float = 0.0
        self._prev_invalid_actions: List[TActType] = []
        self._invalid_actions: List[TActType] = []

        # tracking
        self._tracking_data: List[Dict[str, Any]] = []
        self._tracking_keys: List[str] = []

    def _ready_policy(self):
        """policyの準備を実行。1週目は on_reset、2週目以降は on_step を実行。"""
        # encode -> set invalid -> on_step -> step_reward=0
        logger.debug("ready_policy")

        # --- state
        self._prev_state = self._state
        state = cast(TObsType, self._config.state_encode_one_step(self.env.state, self._env))
        if self._use_stacked_state:
            del self._one_states[0]
            self._one_states.append(state)
            state = self._config.observation_space_one_step.encode_stack(self._one_states)
        self._state = state

        # --- render image
        if self._use_render_image:
            self._prev_render_image = self._render_image
            render_image = self._config.render_image_state_encode_one_step(self._env)
            if self._use_stacked_render_image:
                del self._one_render_images[0]
                self._one_render_images.append(render_image)
                render_image = self._config.obs_render_img_space_one_step.encode_stack(self._one_render_images)
            self._render_image = render_image

        # --- invalid_actions
        self._prev_invalid_actions = self._invalid_actions
        self._invalid_actions = [
            cast(TActType, self._config.action_encode(a))
            for a in self._env.get_invalid_actions(self._player_index)  #
        ]

        # --- reset/step
        if not self._is_reset:
            logger.debug("on_reset")
            self._is_reset = True  # backupのために前に代入、on_reset後は状態変化させない
            self._worker.on_reset(self)
        else:
            # reward
            self._reward = self._step_reward
            self._step_reward = 0.0

            self._step_in_episode += 1
            self._step_in_training += 1

            # backupのためにon_step後は状態変化させない
            logger.debug("on_step")
            self._on_step_in_progress = True
            self._worker.on_step(self)
            self._on_step_in_progress = False

    def policy(self) -> EnvActionType:
        self._ready_policy()

        logger.debug("policy")
        self._prev_action = self._action
        self._action = None  # type: ignore
        self._action = self._worker.policy(self)

        # render
        if self._render.rendering:
            self._render.cache_render(worker=self)

        env_action = self._config.action_decode(cast(RLActionType, self._action))
        return env_action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return

        # 相手の番のrewardも加算
        self._step_reward += self._env.rewards[self.player_index]

        if self._env._done != DoneTypes.NONE:
            # 終了後の状態を取得
            self._ready_policy()

            # 終了後のrender情報
            if self._render.rendering and self._config.render_last_step:
                # policyはRLWorker側で
                # try:
                #     action = self._worker.policy(self)
                #     self._prev_action = self._action
                #     self._action = action
                # except Exception:
                #     logger.info(traceback.format_exc())
                #     logger.warning("'policy()' error in termination status (for rendering)")
                try:
                    self._render.cache_render(worker=self)
                except Exception:
                    logger.info(traceback.format_exc())
                    logger.warning("'render()' error in termination status (for rendering)")

    # ------------------------------------
    # invalid, envとはずれるので直接使わない
    # ------------------------------------
    def get_invalid_actions_to_env(self) -> List[EnvActionType]:
        return [self._config.action_decode(cast(RLActionType, a)) for a in self.invalid_actions]

    def get_valid_actions(self) -> List[TActType]:
        return self._config.action_space.get_valid_actions(self.invalid_actions)

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
        return self._render.get_cached_terminal_text(worker=self, **kwargs)

    def render_terminal_text_to_image(self, **kwargs) -> Optional[np.ndarray]:
        if not self._is_reset:  # on_reset前はrenderしない
            return None
        return self._render.get_cached_terminal_text_to_image(worker=self, **kwargs)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if not self._is_reset:  # on_reset前はrenderしない
            return None
        return self._render.get_cached_rgb_array(worker=self, **kwargs)

    def render_rl_image(self) -> Optional[np.ndarray]:
        if not self._config.observation_space_one_step.is_image(in_image=False):
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
        ----------------------------------
        | BG white    | BG black         |
        | [env render]| [info]           |
        | [env]       | [rl render text] |
        | [rl state]  | [rl render rgb]  |
        | [rl render] |                  |
        ----------------------------------
        """
        padding = 1
        border_color = (111, 175, 0)

        # [env render]
        env_img = self._env.render_rgb_array()
        if env_img is None:
            env_img = self._env.render_terminal_text_to_image()
        assert env_img is not None
        env_img = render_funcs.add_padding(env_img, padding, padding, padding, padding, border_color)

        # [env]
        if self._config.observation_space_of_env.is_image(in_image=False):
            space = cast(BoxSpace, self.config.observation_space_of_env)
            if space.check_val(self._env.state):  # render_image等でずれる場合があるので合っている場合のみ処理
                env_state = space.to_image(self._env.state)
                env_state = render_funcs.add_padding(env_state, padding, padding, padding, padding, border_color)
                env_img = render_funcs.vconcat(env_img, env_state)

        # [rl state]
        if self._config.render_rl_image:
            # 同じ場合は省略
            if self.env.observation_space != self.config.observation_space_one_step:
                rl_state_img = self.render_rl_image()
                if rl_state_img is not None:
                    import cv2

                    rl_state_img = cv2.resize(rl_state_img, self._config.render_rl_image_size, interpolation=cv2.INTER_NEAREST)
                    rl_state_img = render_funcs.add_padding(rl_state_img, padding, padding, padding, padding, border_color)
                    env_img = render_funcs.vconcat(env_img, rl_state_img)

        # [rl render]
        if self._config.use_render_image_state():
            import cv2

            rl_render_img = self.config.obs_render_img_space.to_image(self.render_image_state)
            rl_render_img = cv2.resize(rl_render_img, self._config.render_rl_image_size, interpolation=cv2.INTER_NEAREST)
            rl_render_img = render_funcs.add_padding(rl_render_img, padding, padding, padding, padding, border_color)
            env_img = render_funcs.vconcat(env_img, rl_render_img)

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
    def set_tracking_max_size(self, max_size: int = -1):
        """0以下は無制限"""
        if len(self._tracking_data) > max_size:
            for _ in range(max_size - len(self._tracking_data)):
                del self._tracking_data[0]
        self._tracking_size = max_size

    def get_tracking_length(self) -> int:
        return len(self._tracking_data)

    def add_tracking(self, data: Dict[str, Any]):
        if (self._tracking_size > 0) and (len(self._tracking_data) == self._tracking_size):
            del self._tracking_data[0]
        for k in data.keys():
            if k not in self._tracking_keys:
                self._tracking_keys.append(k)
        self._tracking_data.append(data)
        if (self._tracking_size > 0) and (len(self._tracking_data) > self._tracking_size):
            logger.warning("Tracking data size exceeded: %d > %d", len(self._tracking_data), self._tracking_size)

    def get_tracking_data(self) -> List[Dict[str, Any]]:
        return self._tracking_data

    def get_tracking(self, key: str, size: Optional[int] = None, dummy: Any = None) -> List[Any]:
        if size is None:
            return [d[key] if key in d else dummy for d in self._tracking_data]
        if size <= 0:
            return []
        if len(self._tracking_data) < size:
            arr = [d[key] if key in d else dummy for d in self._tracking_data[:]]
            return [dummy for _ in range(size - len(arr))] + arr[:]
        else:
            return [d[key] if key in d else dummy for d in self._tracking_data[-size:]]

    def get_trackings(
        self,
        keys: Optional[List[str]] = None,
        size: int = 0,
        padding_data: dict = {},
        padding_direct: Literal["head", "tail"] = "head",
    ) -> list:
        if keys is None:
            keys = self._tracking_keys
        if size > 0:
            if len(self._tracking_data) < size:
                pad = [
                    [padding_data[k] if k in padding_data else None for k in keys]
                    for _ in range(size - len(self._tracking_data))  #
                ]
                arr = [
                    [d[k] if k in d else None for k in keys]
                    for d in self._tracking_data[:]  #
                ]
                if padding_direct == "head":
                    return pad + arr
                elif padding_direct == "tail":
                    return arr + pad
                else:
                    raise ValueError(padding_direct)
            else:
                return [[d[k] if k in d else None for k in keys] for d in self._tracking_data[-size:]]
        else:
            return [[d[k] if k in d else None for k in keys] for d in self._tracking_data]

    # ------------------------------------
    # backup/restore
    # ------------------------------------
    def backup(self) -> Any:
        logger.debug(f"backup: step={self._step_in_training}")
        d = [
            # setup
            self._is_setup,
            self._step_in_training,
            self._use_stacked_state,
            self._use_render_image,
            self._use_stacked_render_image,
            self._tracking_size,
            # reset
            self._player_index,
            self._episode_seed,
            self._is_reset,
            self._step_in_episode,
            self._config.observation_space.copy_value(self._prev_state),
            self._config.observation_space.copy_value(self._state),
            [self._config.observation_space_one_step.copy_value(s) for s in self._one_states],
            self._config.obs_render_img_space.copy_value(self._prev_render_image) if self._use_render_image else None,
            self._config.obs_render_img_space.copy_value(self._render_image) if self._use_render_image else None,
            [self._config.obs_render_img_space_one_step.copy_value(s) for s in self._one_render_images],
            self._config.action_space.copy_value(self._prev_action),
            self._config.action_space.copy_value(self._action),
            self._step_reward,
            self._reward,
            self._prev_invalid_actions[:],
            self._invalid_actions[:],
            # env
            self._env.backup(),
            # tracking
            [d.copy() for d in self._tracking_data],
        ]

        return d

    def restore(self, dat: Any):
        logger.debug(f"restore: step={dat[0]}")
        # setup
        self._is_setup = dat[0]
        self._step_in_training = dat[1]
        self._use_stacked_state = dat[2]
        self._use_render_image = dat[3]
        self._use_stacked_render_image = dat[4]
        self._tracking_size = dat[5]
        # reset
        self._player_index = dat[6]
        self._episode_seed = dat[7]
        self._is_reset = dat[8]
        self._step_in_episode = dat[9]
        self._prev_state = self._config.observation_space.copy_value(dat[10])
        self._state = self._config.observation_space.copy_value(dat[11])
        self._one_states = [self._config.observation_space.copy_value(s) for s in dat[12]]
        self._prev_render_image = self._config.obs_render_img_space.copy_value(dat[13]) if dat[13] is not None else np.zeros((1,))
        self._render_image = self._config.obs_render_img_space.copy_value(dat[14]) if dat[14] is not None else np.zeros((1,))
        self._one_render_images = [self._config.obs_render_img_space.copy_value(s) for s in dat[15]]
        self._prev_action = self._config.action_space.copy_value(dat[16])
        self._action = self._config.action_space.copy_value(dat[17])
        self._step_reward = dat[18]
        self._reward = dat[19]
        self._prev_invalid_actions = dat[20][:]
        self._invalid_actions = dat[21][:]
        # env
        self._env.restore(dat[22])
        # tracking
        self._tracking_data = [d.copy() for d in dat[23]]

        if self._render.rendering:
            self._render.cache_reset()

    def copy(self) -> "WorkerRun":
        worker = WorkerRun(cast(RLWorkerGeneric, DummyRLWorker(self._config)), self.env)
        dat = self.backup()
        worker.restore(dat)
        return worker

    # ------------------------------------
    # utils
    # ------------------------------------
    def sample_action(self) -> TActType:
        return self._config.action_space.sample(self._invalid_actions)

    def sample_action_to_env(self) -> EnvActionType:
        return self._config.action_space_of_env.sample(self.get_invalid_actions_to_env())

    def override_action(self, env_action: EnvActionType, encode: bool = True) -> TActType:
        if encode:
            rl_action = cast(TActType, self._config.action_encode(env_action))
        else:
            rl_action = cast(TActType, env_action)
        self._action = rl_action
        return rl_action

    def abort_episode(self):
        self._env.abort_episode()

    def print_discrete_action_info(self, maxa: int, func) -> None:
        view_actions_num = min(20, self.config.action_space.n)
        for action in range(view_actions_num):
            if action in self.invalid_actions:
                s = "x"
            elif action == maxa:
                s = "*"
            else:
                s = " "
            rl_s = func(action)

            act_s = self._env.action_to_str(action)
            if act_s == str(action):
                act_s = f"{act_s:3s}"
            else:
                act_s = f"{action}({act_s})"
                act_s = f"{act_s:6s}"
            s += f"{act_s}: {rl_s}"
            print(s)
        if self.config.action_space.n > view_actions_num:
            print("... Some actions have been omitted.")
