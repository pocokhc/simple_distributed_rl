import logging
from typing import Any, Dict, Generic, List, Literal, Optional, cast

import numpy as np

from srl.base.context import RunContext, RunNameTypes
from srl.base.define import DoneTypes, EnvActionType, RenderModeType
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
        return self._render.rendering

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
    def next_state(self) -> TObsType:
        return self._next_state  # type: ignore  # on_step以外はNone

    def get_state_one_step(self, idx: int = -1) -> TObsType:
        return self._one_states[idx] if self._use_stacked_state else self._state

    @property
    def prev_render_image_state(self) -> np.ndarray:
        return self._prev_render_image

    @property
    def render_image_state(self) -> np.ndarray:
        return self._render_image

    @property
    def next_render_image_state(self) -> np.ndarray:
        return self._next_render_image  # type: ignore  # on_step以外はNone

    def get_render_image_state_one_step(self, idx: int = -1) -> np.ndarray:
        return self._one_render_images[idx] if self._use_stacked_render_image else self._render_image

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
    def next_invalid_actions(self) -> List[TActType]:
        return self._next_invalid_actions  # type: ignore  # on_step以外はNone

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
    ):
        if context is None:
            context = RunContext(self.env.config, self._config)
        if render_mode == "":
            render_mode = context.render_mode

        if self._config.used_rgb_array:
            if render_mode == "terminal":
                logger.warning("'rgb_array' is used, but 'terminal' is specified in render_mode.(Both 'rgb_array' and 'teminal' are used)")
            elif render_mode == "rgb_array":
                pass
            else:
                if context.run_name != RunNameTypes.eval:
                    logger.info(f"[{context.flow_mode}] change render_mode: {render_mode} -> rgb_array")
                render_mode = "rgb_array"

        self._setup_val(context)
        self._render.set_render_mode(
            render_mode,
            context.use_rl_terminal,
            context.use_rl_rgb_array,
            enable_window=False,
        )
        logger.debug(f"on_setup: {render_mode=}")
        self._worker.on_setup(self, context)
        self._is_setup = True

    def _setup_val(self, context: RunContext):
        self._context = context
        self._step_in_training: int = 0

        self._use_stacked_state = self._config.window_length > 1
        self._use_render_image = self._config.use_render_image_state()
        self._use_stacked_render_image = self._use_render_image and (self._config.render_image_window_length > 1)

        self._tracking_size = 0  # 0: no tracking, -1: episode_tracking

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

        # --- state
        self._prev_state: TObsType = self._config.observation_space.get_default()
        self._state: TObsType = self._config.observation_space.get_default()
        self._next_state: Optional[TObsType] = None
        self._one_states = [self._config.observation_space_one_step.get_default() for _ in range(self._config.window_length)]

        if self._use_render_image:
            self._prev_render_image: np.ndarray = self._config.obs_render_img_space.get_default()
            self._render_image: np.ndarray = self._config.obs_render_img_space.get_default()
            self._next_render_image: Optional[np.ndarray] = None
            self._one_render_images = [self._config.obs_render_img_space_one_step.get_default() for _ in range(self._config.render_image_window_length)]
        else:
            self._prev_render_image: np.ndarray = np.zeros((1,))
            self._render_image: np.ndarray = np.zeros((1,))
            self._next_render_image: Optional[np.ndarray] = None
            self._one_render_images = []

        # action, reward, done
        self._action = self._config.action_space.get_default()
        self._step_reward: float = 0.0
        self._reward: float = 0.0
        self._prev_invalid_actions: List[TActType] = []
        self._invalid_actions: List[TActType] = []
        self._next_invalid_actions: Optional[List[TActType]] = None

        # tracking
        self._tracking_data: List[Dict[str, Any]] = []

    def _ready_policy(self):
        """policyの準備を実行。1週目は on_reset、2週目以降は on_step を実行。"""
        # encode -> set invalid -> on_step -> step_reward=0

        # --- state
        state = cast(TObsType, self._config.state_encode_one_step(self.env.state, self._env))
        if self._use_stacked_state:
            del self._one_states[0]
            self._one_states.append(state)
            state = self._config.observation_space_one_step.encode_stack(self._one_states)

        # --- render image
        if self._use_render_image:
            render_image = self._config.render_image_state_encode_one_step(self._env)
            if self._use_stacked_render_image:
                del self._one_render_images[0]
                self._one_render_images.append(render_image)
                render_image = self._config.obs_render_img_space_one_step.encode_stack(self._one_render_images)

        # --- invalid_actions
        invalid_actions = [
            cast(TActType, self._config.action_encode(a))
            for a in self._env.get_invalid_actions(self._player_index)  #
        ]

        # --- reset/step
        if not self._is_reset:
            # state
            self._state = state
            if self._use_render_image:
                self._render_image = render_image

            # invalid_actions
            self._invalid_actions = invalid_actions

            logger.debug("on_reset")
            self._worker.on_reset(self)
            self._is_reset = True
        else:
            # update next
            self._next_state = state
            if self._use_render_image:
                self._next_render_image = render_image
            self._next_invalid_actions = invalid_actions

            # reward
            self._reward = self._step_reward
            self._step_reward = 0.0

            logger.debug("on_step")
            self._worker.on_step(self)
            self._step_in_episode += 1
            self._step_in_training += 1

            # step後にupdate
            self._prev_state = self._state
            self._state = self._next_state
            self._next_state = None
            if self._use_render_image:
                self._prev_render_image = self._render_image
                assert self._next_render_image is not None
                self._render_image = self._next_render_image
                self._next_render_image = None
            self._prev_invalid_actions = self._invalid_actions
            self._invalid_actions = self._next_invalid_actions
            self._next_invalid_actions = None

    def policy(self) -> EnvActionType:
        self._ready_policy()

        logger.debug("policy")
        action = self._worker.policy(self)
        self._action = action

        # render
        if self._render.rendering:
            self._render.cache_render(worker=self)

        env_action = self._config.action_decode(action)
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

    # ------------------------------------
    # invalid, envとはずれるので直接使わない
    # ------------------------------------
    def get_invalid_actions_to_env(self) -> List[EnvActionType]:
        return [self._config.action_decode(a) for a in self.invalid_actions]

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
            if self.env.observation_space != self.config._rl_obs_space_one_step:
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
        """-1 は無制限"""
        if len(self._tracking_data) > max_size:
            for _ in range(max_size - len(self._tracking_data)):
                del self._tracking_data[0]
        self._tracking_size = max_size

    def get_tracking_length(self) -> int:
        return len(self._tracking_data)

    def add_tracking(self, data: Dict[str, Any]):
        if (self._tracking_size > 0) and (len(self._tracking_data) == self._tracking_size):
            del self._tracking_data[0]
        self._tracking_data.append(data)

    def get_tracking(self, key: str, size: int = -1, dummy: Any = None) -> List[Any]:
        if size > 0:
            if len(self._tracking_data) < size:
                arr = [d[key] if key in d else dummy for d in self._tracking_data[:]]
                return [dummy for _ in range(size - len(arr))] + arr[:]
            else:
                return [d[key] if key in d else dummy for d in self._tracking_data[-size:]]
        return [d[key] if key in d else dummy for d in self._tracking_data]

    def get_trackings(
        self,
        keys: List[str] = [],
        size: int = 0,
        padding_data: dict = {},
        padding_direct: Literal["head", "tail"] = "head",
    ) -> list:
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
                return [[d[k] for k in keys] for d in self._tracking_data[-size:]]
        else:
            return [[d[k] for k in keys] for d in self._tracking_data]

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
            self._config.observation_space.copy_value(self._next_state) if self._next_state is not None else None,
            [self._config.observation_space_one_step.copy_value(s) for s in self._one_states],
            self._config.obs_render_img_space.copy_value(self._prev_render_image),
            self._config.obs_render_img_space.copy_value(self._render_image),
            self._config.obs_render_img_space.copy_value(self._next_render_image) if self._next_render_image is not None else None,
            [self._config.obs_render_img_space_one_step.copy_value(s) for s in self._one_render_images],
            self._config.action_space.copy_value(self._action),
            self._step_reward,
            self._reward,
            self._prev_invalid_actions[:],
            self._invalid_actions[:],
            self._next_invalid_actions[:] if self._next_invalid_actions is not None else None,
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
        self._prev_state = dat[10]
        self._state = dat[11]
        self._next_state = dat[12] if dat[12] is not None else None
        self._one_states = dat[13][:]
        self._prev_render_image = dat[14]
        self._render_image = dat[15]
        self._next_render_image = dat[16] if dat[16] is not None else None
        self._one_render_images = dat[17][:]
        self._action = dat[18]
        self._step_reward = dat[19]
        self._reward = dat[20]
        self._prev_invalid_actions = dat[21][:]
        self._invalid_actions = dat[22][:]
        self._next_invalid_actions = dat[23][:] if dat[23] is not None else None
        # env
        self._env.restore(dat[24])
        # tracking
        self._tracking_data = dat[25][:]

        if self._render.rendering:
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

    def print_discrete_action_info(self, maxa: int, func) -> None:
        view_actions_num = min(15, self.config.action_space.n)
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
