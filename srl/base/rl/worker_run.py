import logging
from typing import Any, Dict, Generic, List, Optional, Union, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import (
    DoneTypes,
    EnvActionType,
    RenderModes,
    SpaceTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.exception import SRLError
from srl.base.info import Info
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import DummyRLWorker, RLWorkerGeneric
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
        return self._tracking_stacked_states[-2] if self._use_stacked_state else self._tracking_one_states[-2]

    @property
    def state(self) -> TObsType:
        return self._tracking_stacked_states[-1] if self._use_stacked_state else self._tracking_one_states[-1]

    @property
    def state_one_step(self) -> TObsType:
        return self._tracking_one_states[-1]

    @property
    def prev_render_image_state(self) -> np.ndarray:
        return self._tracking_stacked_render_images[-2] if self._use_stacked_render_image else self._tracking_one_render_images[-2]

    @property
    def render_image_state(self) -> np.ndarray:
        return self._tracking_stacked_render_images[-1] if self._use_stacked_render_image else self._tracking_one_render_images[-1]

    @property
    def render_image_state_one_step(self) -> np.ndarray:
        return self._tracking_one_render_images[-1]

    @property
    def prev_action(self) -> TActType:
        return self._tracking_action[-2]

    def get_onehot_prev_action(self):
        return self._config.action_space.get_onehot(self._tracking_action[-2])

    @property
    def action(self) -> TActType:
        return self._tracking_action[-1]

    def get_onehot_action(self, action=None):
        return self._config.action_space.get_onehot(self._tracking_action[-1] if action is None else action)

    @property
    def reward(self) -> float:
        return self._tracking_reward[-1]

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
        return self._tracking_invalid_actions[-2]

    @property
    def invalid_actions(self) -> List[TActType]:
        return self._tracking_invalid_actions[-1]

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

        # prev用に2は保持する
        self._tracking_episode = False
        self._tracking_size = 2
        self._tracking_one_state_size = max(2, self._config.window_length)
        self._use_stacked_state = self._config.window_length > 1
        # render image
        self._use_render_image = self._config.use_render_image_state()
        self._use_stacked_render_image = self._use_render_image and (self._config.render_image_window_length > 1)
        self._tracking_one_render_image_size = max(2, self._config.render_image_window_length)

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
        self._tracking_one_states: List[TObsType] = [
            self._config.observation_space_one_step.get_default()
            for _ in range(self._tracking_one_state_size)  #
        ]
        if self._use_stacked_state:
            self._tracking_stacked_states: List[TObsType] = [
                self._config.observation_space.get_default()
                for _ in range(2)  #
            ]
        if self._use_render_image:
            self._tracking_one_render_images: List[np.ndarray] = [
                self._config.obs_render_img_space_one_step.get_default()
                for _ in range(self._tracking_one_render_image_size)  #
            ]
            if self._use_stacked_render_image:
                self._tracking_stacked_render_images: List[np.ndarray] = [
                    self._config.obs_render_img_space.get_default()
                    for _ in range(2)  #
                ]

        # action
        self._tracking_action: List[TActType] = [
            self._config.action_space.get_default()
            for _ in range(2)  #
        ]
        # reward, done
        self._tracking_reward: List[float] = [0.0]
        self._tracking_terminate: List[int] = []
        self._tracking_invalid_actions: List[List[TActType]] = [[], []]
        self._tracking_user_data: List[Dict[str, Any]] = []

        self._step_reward: float = 0

    def ready_policy(self):
        """policyの準備を実行。1週目は on_reset、2週目以降は on_step を実行。"""
        # encode -> set invalid -> on_step -> step_reward=0
        if self._is_ready_policy:
            raise SRLError("'ready_policy' cannot be called consecutively.")

        # --- state
        state = cast(TObsType, self._config.state_encode_one_step(self.env.state, self._env))
        if len(self._tracking_one_states) == self._tracking_one_state_size:
            del self._tracking_one_states[0]
        self._tracking_one_states.append(state)
        if self._use_stacked_state:
            stacked_state = self._config.observation_space_one_step.encode_stack(
                self._tracking_one_states[-self._config.window_length :],
            )
            if len(self._tracking_stacked_states) == self._tracking_size:
                del self._tracking_stacked_states[0]
            self._tracking_stacked_states.append(stacked_state)
        if self._use_render_image:
            render_image = self._config.render_image_state_encode_one_step(self._env)
            if len(self._tracking_one_render_images) == self._tracking_one_render_image_size:
                del self._tracking_one_render_images[0]
            self._tracking_one_render_images.append(render_image)
            if self._use_stacked_render_image:
                stacked_img = self._config.obs_render_img_space_one_step.encode_stack(
                    self._tracking_one_render_images[-self._config.render_image_window_length :],
                )
                if len(self._tracking_stacked_render_images) == self._tracking_size:
                    del self._tracking_stacked_render_images[0]
                self._tracking_stacked_render_images.append(stacked_img)

        # invalid_actions
        if len(self._tracking_invalid_actions) == self._tracking_size:
            del self._tracking_invalid_actions[0]
        self._tracking_invalid_actions.append(
            [
                cast(TActType, self._config.action_encode(a))
                for a in self._env.get_invalid_actions(self._player_index)  #
            ]
        )

        if not self._is_reset:
            logger.debug("on_reset")
            self._worker.on_reset(self)
            self._is_reset = True
        else:
            if self._tracking_episode:
                if len(self._tracking_reward) == self._tracking_size:
                    del self._tracking_reward[0]
                self._tracking_reward.append(self._step_reward)
                if len(self._tracking_terminate) == self._tracking_size:
                    del self._tracking_terminate[0]
                self._tracking_terminate.append(int(self.terminated))
            else:
                self._tracking_reward[0] = self._step_reward

            self._step_reward = 0.0
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
            return self._tracking_invalid_actions[-1]
        return [
            cast(TActType, self._config.action_encode(a))
            for a in env.get_invalid_actions(player_index)  #
        ]

    def get_valid_actions(self, env: Optional[EnvRun] = None, player_index: int = -1) -> List[TActType]:
        return self._config.action_space.get_valid_actions(self.get_invalid_actions(env, player_index))

    def add_invalid_actions(self, invalid_actions: List[TActType], encode: bool = False) -> None:
        if encode:
            invalid_actions = [self._config.action_encode(a) for a in invalid_actions]  # type: ignore
        self._tracking_invalid_actions[-1] = list(set(self._tracking_invalid_actions[-1] + invalid_actions))

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
        if not self._config._rl_obs_space_one_step.is_image():
            return None
        img = cast(np.ndarray, self._tracking_one_states[-1]).copy()
        if img.max() <= 1:
            img *= 255
        if self._config._rl_obs_space_one_step.stype == SpaceTypes.GRAY_2ch:
            img = img[..., np.newaxis]
            img = np.tile(img, (1, 1, 3))
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.GRAY_3ch:
            img = np.tile(img, (1, 1, 3))
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.COLOR:
            pass
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.IMAGE:
            if len(img.shape) == 3 and img.shape[-1] == 3:
                pass
            else:
                return None
        else:
            return None
        return img.astype(np.uint8)

    def create_render_image(
        self,
        add_terminal: bool = True,
        add_rgb_array: bool = True,
        add_rl_state: bool = True,
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
        if add_rl_state:
            # 同じ場合は省略
            if self.env.observation_space != self.config._rl_obs_space_one_step:
                rl_state_img = self.render_rl_image()
                if rl_state_img is not None:
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
    def enable_tracking(self, max_size: int = 9999):
        """stateだけ max_size+1 確保"""
        if self._is_setup:
            raise SRLError("Please call it in 'on_setup()'")
        assert max_size > 0
        self._tracking_episode = True
        self._tracking_size = max(self._tracking_size, max_size)
        self._tracking_one_state_size = max(self._tracking_one_state_size, max_size + 1)
        if self._use_render_image and self._use_stacked_render_image:
            self._tracking_one_render_image_size = max(self._tracking_one_render_image_size, max_size + 1)

    def _check_tracking_key(self, data: Dict[str, Any]):
        for k in data.keys():
            if k in [
                "state",
                "one_state",
                "stacked_state",
                "render_image",
                "one_render_image",
                "stacked_render_image",
                "action",
                "invalid_actions",
                "reward",
                "terminated",
            ]:
                raise ValueError(f"'{k}' is a reserved word.")

    def update_tracking(self, data: Dict[str, Any], index: int = -1):
        self._check_tracking_key(data)
        self._tracking_user_data[index].update(data)

    def add_tracking(self, data: Dict[str, Any]):
        self._check_tracking_key(data)
        if len(self._tracking_user_data) == self._tracking_size:
            del self._tracking_user_data[0]
        self._tracking_user_data.append(data)

    def get_tracking(self, key: str, size: int = 0, dummy: Any = None) -> List[Any]:
        if size <= 0:
            size = self._tracking_size

        if key == "state":
            if self._use_stacked_state:
                arr = self._tracking_stacked_states
                if dummy is None:
                    dummy = self._config.observation_space.get_default()
            else:
                arr = self._tracking_one_states
                if dummy is None:
                    dummy = self._config.observation_space_one_step.get_default()
        elif key == "one_state":
            arr = self._tracking_one_states
            if dummy is None:
                dummy = self._config.observation_space_one_step.get_default()
        elif key == "stacked_state":
            arr = self._tracking_stacked_states
            if dummy is None:
                dummy = self._config.observation_space.get_default()
        elif key == "render_image":
            if self._use_stacked_render_image:
                arr = self._tracking_stacked_render_images
                if dummy is None:
                    dummy = self._config.obs_render_img_space.get_default()
            else:
                arr = self._tracking_one_render_images
                if dummy is None:
                    dummy = self._config.obs_render_img_space_one_step.get_default()
        elif key == "one_render_image":
            arr = self._tracking_one_render_images
            if dummy is None:
                dummy = self._config.obs_render_img_space_one_step.get_default()
        elif key == "stacked_render_image":
            arr = self._tracking_stacked_render_images
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
            if len(self._tracking_user_data) < size:
                arr = [d[key] if key in d else dummy for d in self._tracking_user_data[:]]
                return [dummy for _ in range(size - len(arr))] + arr[:]
            else:
                return [d[key] if key in d else dummy for d in self._tracking_user_data[-size:]]

        if len(arr) < size:
            return [dummy for _ in range(size - len(arr))] + arr[:]
        else:
            return arr[-size:]

    def add_dummy_step(self, state=None, action=None, invalid_actions=[], reward: float = 0.0, terminated: bool = False, tracking_data: Dict[str, Any] = {}, is_reset: bool = False):
        self._check_tracking_key(tracking_data)

        # reset時はstateの挙動は複雑なので何もしない
        if is_reset and (state is not None):
            logger.warning("No state is added when resetting.")
        if not is_reset:
            if state is None:
                state = self._config.observation_space_one_step.get_default()
            if len(self._tracking_one_states) == self._tracking_one_state_size:
                del self._tracking_one_states[0]
            self._tracking_one_states.append(state)
            if self._use_stacked_state:
                stacked_state = self._config.observation_space_one_step.encode_stack(
                    self._tracking_one_states[-self._config.window_length :],
                )
                if len(self._tracking_stacked_states) == self._tracking_size:
                    del self._tracking_stacked_states[0]
                self._tracking_stacked_states.append(stacked_state)
            if self._use_render_image:
                render_image = self._config.render_image_state_encode_one_step(self._env)
                if len(self._tracking_one_render_images) == self._tracking_one_render_image_size:
                    del self._tracking_one_render_images[0]
                self._tracking_one_render_images.append(render_image)
                if self._use_stacked_render_image:
                    stacked_img = self._config.obs_render_img_space_one_step.encode_stack(
                        self._tracking_one_render_images[-self._config.render_image_window_length :],
                    )
                    if len(self._tracking_stacked_render_images) == self._tracking_size:
                        del self._tracking_stacked_render_images[0]
                    self._tracking_stacked_render_images.insert(-1, stacked_img)

        # action
        if action is None:
            action = self._config.action_space.get_default()
        if len(self._tracking_action) == self._tracking_size:
            del self._tracking_action[0]
        self._tracking_action.append(action)

        # invalid action
        if len(self._tracking_invalid_actions) == self._tracking_size:
            del self._tracking_invalid_actions[0]
        self._tracking_invalid_actions.append(invalid_actions)

        # reward
        if len(self._tracking_reward) == self._tracking_size:
            del self._tracking_reward[0]
        self._tracking_reward.append(reward)

        # terminate
        if len(self._tracking_terminate) == self._tracking_size:
            del self._tracking_terminate[0]
        self._tracking_terminate.append(int(terminated))

        # user
        if len(self._tracking_user_data) == self._tracking_size:
            del self._tracking_user_data[0]
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
            self._tracking_episode,
            self._tracking_size,
            self._tracking_one_state_size,
            self._use_stacked_state,
            self._use_render_image,
            self._use_stacked_render_image,
            self._tracking_one_render_image_size,
            # reset
            self._player_index,
            self._episode_seed,
            self._is_reset,
            self._is_ready_policy,
            [self._config.observation_space_one_step.copy_value(s) for s in self._tracking_one_states],
            ([self._config.observation_space.copy_value(s) for s in self._tracking_stacked_states] if self._use_stacked_state else []),
            ([self._config.obs_render_img_space_one_step.copy_value(s) for s in self._tracking_one_render_images] if self._use_render_image else []),
            ([self._config.obs_render_img_space.copy_value(s) for s in self._tracking_stacked_render_images] if self._use_stacked_render_image else []),
            [self._config.action_space.copy_value(d) for d in self._tracking_action],
            self._tracking_reward[:],
            self._tracking_terminate[:],
            self._tracking_invalid_actions[:],
            [d.copy() for d in self._tracking_user_data],
            self._step_reward,
            # env
            self._env.backup(),
        ]
        return d

    def restore(self, dat: Any):
        logger.debug(f"restore: step={dat[0]}")
        # setup
        self._is_setup = dat[0]
        self._total_step = dat[1]
        self._tracking_episode = dat[2]
        self._tracking_size = dat[3]
        self._tracking_one_state_size = dat[4]
        self._use_stacked_state = dat[5]
        self._use_render_image = dat[6]
        self._use_stacked_render_image = dat[7]
        self._tracking_one_render_image_size = dat[8]
        # reset
        self._player_index = dat[9]
        self._episode_seed = dat[10]
        self._is_reset = dat[11]
        self._is_ready_policy = dat[12]
        self._tracking_one_states = dat[13][:]
        self._tracking_stacked_states = dat[14][:]
        self._tracking_one_render_images = dat[15][:]
        self._tracking_stacked_render_images = dat[16][:]
        self._tracking_action = dat[17][:]
        self._tracking_reward = dat[18][:]
        self._tracking_terminate = dat[19][:]
        self._tracking_invalid_actions = dat[20][:]
        self._tracking_user_data = dat[21][:]
        self._step_reward = dat[22]
        # env
        self._env.restore(dat[23])

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
