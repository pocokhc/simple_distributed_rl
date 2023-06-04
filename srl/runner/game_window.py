import enum
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pygame

import srl
from srl.base.define import EnvActionType, KeyBindType, PlayRenderModes
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.spaces.box import BoxSpace
from srl.runner.callback import GameCallback
from srl.runner.callbacks.history_episode import HistoryEpisode
from srl.utils import pygame_wrapper as pw

logger = logging.getLogger(__name__)


class KeyStatus(enum.Enum):
    UP = enum.auto()  # 離している間
    DOWN = enum.auto()  # 押している間
    PRESSED = enum.auto()  # 押した瞬間のみ
    RELEASED = enum.auto()  # 離した瞬間のみ


class _GameWindow(ABC):
    def __init__(self, _is_test: bool = False) -> None:
        self._is_test = _is_test  # for test

        self.title: str = ""
        self.padding: int = 4
        self.img_dir = os.path.join(os.path.dirname(__file__), "img")
        self.keys_status = {}

        self.org_env_w = 0
        self.org_env_h = 0
        self.rl_w = 0
        self.rl_h = 0
        self.info_w = 100
        self.info_h = 100
        self.scale = 1
        self.resize(1.0)

    @abstractmethod
    def on_loop(self, events: List[pygame.event.Event]):
        raise NotImplementedError()

    def get_key(self, key) -> KeyStatus:
        return self.keys_status.get(key, KeyStatus.UP)

    def get_down_keys(self) -> List[int]:
        return [k for k, s in self.keys_status.items() if s == KeyStatus.DOWN]

    def get_pressed_keys(self) -> List[int]:
        return [k for k, s in self.keys_status.items() if s == KeyStatus.PRESSED]

    def play(self):
        if "SDL_VIDEODRIVER" in os.environ:
            pygame.display.quit()
            del os.environ["SDL_VIDEODRIVER"]

        # --- pygame init
        self.base_info_x = self.env_w + self.padding
        self.base_info_y = self.padding

        pygame.init()
        pygame.display.set_caption(self.title)
        self.resize(1.0)
        clock = pygame.time.Clock()
        pygame.key.set_repeat(500, 30)

        # -------------------------------
        # pygame window loop
        # -------------------------------
        pygame_done = False
        while not pygame_done:
            # --- key check
            for k in self.keys_status.keys():
                if self.keys_status[k] == KeyStatus.RELEASED:
                    self.keys_status[k] = KeyStatus.UP
                if self.keys_status[k] == KeyStatus.PRESSED:
                    self.keys_status[k] = KeyStatus.DOWN

            # --- event check
            is_window_resize = False
            _valid_unicode_keys = ["-", "+"]
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame_done = True
                elif event.type == pygame.KEYUP:
                    self.keys_status[event.key] = KeyStatus.RELEASED
                    if event.unicode in _valid_unicode_keys:
                        self.keys_status[event.unicode] = KeyStatus.RELEASED

                elif event.type == pygame.KEYDOWN:
                    if self.keys_status.get(event.key, KeyStatus.UP) != KeyStatus.DOWN:
                        self.keys_status[event.key] = KeyStatus.PRESSED
                    if event.unicode in _valid_unicode_keys:
                        if self.keys_status.get(event.unicode, KeyStatus.UP) != KeyStatus.DOWN:
                            self.keys_status[event.unicode] = KeyStatus.PRESSED

                    if event.key == pygame.K_ESCAPE:
                        pygame_done = True
                    elif event.unicode == "1":
                        self.scale = 0.5
                        is_window_resize = True
                    elif event.unicode == "2":
                        self.scale = 1.0
                        is_window_resize = True
                    elif event.unicode == "3":
                        self.scale = 1.5
                        is_window_resize = True
                    elif event.unicode == "4":
                        self.scale = 2.0
                        is_window_resize = True
                    elif event.unicode == "5":
                        self.scale = 3.0
                        is_window_resize = True
                    elif event.unicode == "6":
                        self.scale = 4.0
                        is_window_resize = True

            # --- window check
            if self.org_env_w < self.env_image.shape[1]:
                self.org_env_w = self.env_image.shape[1]
                is_window_resize = True
            if self.org_env_h < self.env_image.shape[0]:
                self.org_env_h = self.env_image.shape[0]
                is_window_resize = True
            if self.rl_w < self.rl_image.shape[1]:
                self.rl_w = self.rl_image.shape[1]
                is_window_resize = True
            if self.rl_h < self.rl_image.shape[0]:
                self.rl_h = self.rl_image.shape[0]
                is_window_resize = True

            self.screen.fill((0, 0, 0))

            # --- image
            pw.draw_image_rgb_array(
                self.screen,
                0,
                0,
                self.env_image,
                resize=(
                    int(self.env_image.shape[1] * self.scale),
                    int(self.env_image.shape[0] * self.scale),
                ),
            )
            pw.draw_image_rgb_array(self.screen, self.base_rl_x, self.base_rl_y, self.rl_image)

            # --- info
            self.info_texts = []
            self.hotkey_texts = [
                "- Hotkeys -",
                f"1-6: change screen size (x{self.scale:.1f})",
            ]

            self.on_loop(events)
            self.info_texts.append("")
            self.info_texts.extend(self.hotkey_texts)
            width, height = pw.draw_texts(
                self.screen,
                self.base_info_x,
                self.base_info_y,
                self.info_texts,
                size=16,
                color=(255, 255, 255),
            )
            if self.info_w < width:
                self.info_w = width
                is_window_resize = True
            if self.info_h < height:
                self.info_h = height
                is_window_resize = True

            pygame.display.flip()
            clock.tick(60)

            if is_window_resize:
                self.resize(self.scale)

            if self._is_test:
                pygame_done = True

    def set_image(self, env_image: np.ndarray, rl_image: Optional[np.ndarray]):
        if rl_image is None:
            rl_image = np.zeros((0, 0, 3))
        self.env_image = env_image
        self.rl_image = rl_image

    def add_hotkey_texts(self, texts: List[str]):
        self.hotkey_texts.extend(texts)

    def add_info_texts(self, texts: List[str]):
        self.info_texts.extend(texts)

    def resize(self, scale: float):
        self.scale = scale
        self.env_w = self.org_env_w * scale
        self.env_h = self.org_env_h * scale

        self.base_rl_x = self.env_w + self.padding
        self.base_rl_y = self.padding
        self.base_info_x = self.base_rl_x + self.rl_w + self.padding
        self.base_info_y = self.padding

        window_w = self.env_w + self.padding + self.rl_w + self.padding + self.info_w
        window_h = max(max(self.env_h, self.rl_h), self.info_h) + self.padding * 2

        window_w = min(window_w, 1900)
        window_h = min(window_h, 1600)
        self.screen = pygame.display.set_mode((window_w, window_h))

    def draw_texts(
        self,
        x: float,
        y: float,
        texts: List[str],
        size: int = 16,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        pw.draw_texts(self.screen, x, y, texts, color=color, size=size)


class PlayableGame(_GameWindow):
    def __init__(
        self,
        env_config: Union[str, EnvConfig],
        # players: List[Union[None, str, RLConfig]] = [],  TODO
        key_bind: KeyBindType = None,
        action_division_num: int = 5,
        rl_config: Optional[RLConfig] = None,
        rl_config_player: int = 0,  # rl_config_player が記録するプレイヤー
        callbacks: List[GameCallback] = [],
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)
        self.env = srl.make_env(env_config)
        self.action_division_num = action_division_num
        self.callbacks = callbacks[:]
        self.noop = None
        self.step_time = 0
        self.rl_config = rl_config
        self.rl_config_player = rl_config_player

        # --- key bind (扱いやすいように変形)
        if key_bind is None:
            key_bind = self.env.get_key_bind()
        if key_bind is None:
            self.key_bind = None
        else:
            self.key_bind = {}
            self.key_bind_str = {}
            for key_combination, action in key_bind.items():
                if key_combination == "":
                    self.noop = action
                    continue
                if isinstance(key_combination, int):
                    key_combination = [key_combination]

                # key_bind
                key_code = tuple(sorted(ord(key) if isinstance(key, str) else key for key in key_combination))
                self.key_bind[key_code] = action

                # key_str
                key_names = [pygame.key.name(key) if isinstance(key, int) else str(key) for key in key_combination]
                self.key_bind_str[",".join(key_names)] = action

        self.key_bind = cast(Dict[Optional[Tuple[int]], EnvActionType], self.key_bind)

        # --- reset
        self.env.reset(render_mode=PlayRenderModes.rgb_array)
        self.env_interval = self.env.config.render_interval
        self.set_image(self.env.render_rgb_array(), None)
        if rl_config is not None:
            assert 0 <= rl_config_player < self.env.player_num
            self.remote_memory = srl.make_remote_memory(rl_config, self.env)
            parameter = srl.make_parameter(rl_config, self.env)
            self.rl_worker = srl.make_worker(rl_config, parameter, self.remote_memory, training=True)
            self.rl_worker.on_reset(self.env, self.rl_config_player)
        else:
            self.remote_memory = None

        self.scene = "START"
        self.mode = "Turn"  # "Turn" or "RealTime"
        self.is_pause = False
        self.action = 0
        self.cursor_action = 0
        self.valid_actions = []

        self._callback_info = {
            "env": self.env,
        }
        [c.on_game_init(self._callback_info) for c in self.callbacks]

    def _env_step(self, action):
        t0 = time.time()

        # worker.policy
        if self.rl_config is not None:
            _ = self.rl_worker.policy(self.env)

        # env.step
        self.env.step(action)

        # worker.on_step
        if self.rl_config is not None:
            self.rl_worker.on_step(self.env)

        # render
        self.set_image(self.env.render_rgb_array(), None)
        invalid_actions = self.env.get_invalid_actions()
        self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
        if self.cursor_action >= len(self.valid_actions):
            self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.env.action_space.decode_from_int(self.action)

        self._callback_info["action"] = action
        [c.on_game_step_end(self._callback_info) for c in self.callbacks]

        if self.env.done:
            self.scene = "START"
            [c.on_game_end(self._callback_info) for c in self.callbacks]

        self.step_time = time.time() - t0

    def on_loop(self, events: List[pygame.event.Event]):
        # --- 全体
        t = []
        t.append("r  : Reset")
        if self.mode == "RealTime":
            t.append(f"-+ : change speed ({self.env_interval:.0f}ms; {1000/self.env_interval:.1f}fps)")
            if self.is_pause:
                t.append("p  : pause/unpause (Pause)")
            else:
                t.append("p  : pause/unpause (UnPause)")
            t.append("f  : FrameAdvance")
        self.add_hotkey_texts(t)

        if self.mode == "RealTime":
            if self.get_key("-") == KeyStatus.PRESSED:
                self.env_interval *= 2
            elif self.get_key("+") == KeyStatus.PRESSED:
                self.env_interval /= 2
                if self.env_interval < 1:
                    self.env_interval = 1
            elif self.get_key(pygame.K_p) == KeyStatus.PRESSED:
                self.is_pause = not self.is_pause

        if self.scene == "START":
            # --- START
            if self.get_key(pygame.K_UP) == KeyStatus.PRESSED:
                self.mode = "Turn"
            elif self.get_key(pygame.K_DOWN) == KeyStatus.PRESSED:
                self.mode = "RealTime"
            elif self.get_key(pygame.K_RETURN) == KeyStatus.PRESSED:
                self.scene = "RESET"

            if self.mode == "Turn":
                self.add_info_texts(["> Turn", "  RealTime"])
            else:
                self.add_info_texts(["  Turn", "> RealTime"])

        elif self.scene == "RUNNING":
            # --- RUNNING
            if self.get_key(pygame.K_r) == KeyStatus.PRESSED:
                self.scene = "START"
            elif (self.mode == "RealTime") and (self.get_key(pygame.K_f) == KeyStatus.PRESSED):
                self.frameadvance = True
                self.is_pause = True
            elif self.key_bind is None:
                # key_bindがない場合のアクションを決定
                if self.get_key(pygame.K_LEFT) == KeyStatus.PRESSED:
                    self.cursor_action -= 1
                    if self.cursor_action < 0:
                        self.cursor_action = 0
                    self.action = self.valid_actions[self.cursor_action]
                    self.action = self.env.action_space.decode_from_int(self.action)
                elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
                    self.cursor_action += 1
                    if self.cursor_action >= len(self.valid_actions):
                        self.cursor_action = len(self.valid_actions) - 1
                    self.action = self.valid_actions[self.cursor_action]
                    self.action = self.env.action_space.decode_from_int(self.action)

                if self.mode == "Turn":
                    # key_bindがない、Turnはアクション決定で1frame進める
                    is_step = False
                    for event in events:
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN:
                                is_step = True
                                break
                            elif event.key == pygame.K_z:
                                is_step = True
                                break
                            elif event.key == pygame.K_f:
                                is_step = True
                                break
                    if is_step:
                        self._env_step(self.action)
                        self.action = self.valid_actions[self.cursor_action]
                        self.action = self.env.action_space.decode_from_int(self.action)

            elif self.mode == "Turn":
                # keybindがあり、Turnの場合は押したら進める
                key = tuple(sorted(self.get_pressed_keys()))
                if key in self.key_bind:
                    self.action = self.key_bind[key]
                    self._env_step(self.action)

            if self.mode == "RealTime":
                if self.key_bind is not None:
                    # 押してあるkeys
                    key = tuple(sorted(self.get_down_keys()))
                    if key in self.key_bind:
                        self.action = self.key_bind[key]
                    else:
                        self.action = self.noop
                if self.is_pause:
                    if self.frameadvance:
                        self._env_step(self.action)
                        self.frameadvance = False
                elif time.time() - self.step_t0 > self.env_interval / 1000:
                    self.step_t0 = time.time()
                    self._env_step(self.action)

            self.add_info_texts([f"Select Action {self.valid_actions}"])

            s = " "
            s1 = str(self.action)
            s2 = self.env.action_to_str(self.action)
            if s1 == s2:
                s += s1
            else:
                s += f"{s1}({s2})"
            self.add_info_texts([s])

        # --- key bind
        if self.key_bind is not None:
            t = ["", "- ActionKeys -"]
            if self.noop is not None:
                t.append(f"no: {self.noop}")
            for key, val in self.key_bind_str.items():
                s1 = str(val)
                s2 = self.env.action_to_str(val)
                if s1 == s2:
                    s = s1
                else:
                    s = f"{s1}({s2})"
                t.append(f"{key}: {s}")
            self.add_info_texts(t)

        # --- step info
        s = [
            "",
            "- env infos -",
            f"action_space     : {self.env.action_space}",
            f"observation_type : {self.env.observation_type.name}",
            f"observation_space: {self.env.observation_space}",
            f"player_num       : {self.env.player_num}",
            f"step   : {self.env.step_num}",
            f"next   : {self.env.next_player_index}",
            f"rewards: {self.env.step_rewards}",
            f"info   : {self.env.info}",
            f"done   : {self.env.done}({self.env.done_reason})",
            f"time   : {self.step_time*1000:.1f}ms",
        ]
        self.add_info_texts(s)

        # --- RESET は最後
        if self.scene == "RESET":
            self.scene = "RUNNING"
            if isinstance(self.env.action_space, BoxSpace):
                self.env.action_space.create_division_tbl(self.action_division_num)
            if self.noop is None:
                self.noop = self.env.action_space.decode_from_int(0)
            self.env.reset()
            self.set_image(self.env.render_rgb_array(), None)
            self.action_size = self.env.action_space.n
            invalid_actions = self.env.get_invalid_actions()
            self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
            if self.cursor_action >= len(self.valid_actions):
                self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.env.action_space.decode_from_int(self.action)

            self.step_t0 = time.time()
            if self.mode == "Turn":
                self.frameadvance = False
                self.is_pause = True
            else:
                self.frameadvance = False
                self.is_pause = False

            [c.on_game_begin(self._callback_info) for c in self.callbacks]


class EpisodeReplay(_GameWindow):
    def __init__(self, history: HistoryEpisode, _is_test: bool = False) -> None:
        super().__init__(_is_test=_is_test)

        self.history = history
        self.episode = 0
        self.interval = 200  # TODO
        self._set_episode()

    def _set_episode(self):
        self.episode_info, self.episode_data = self.history.load_episode(self.episode)
        self.env_pause = True
        self.step = 0
        self._set_image(0)

    def _set_image(self, step: int):
        env_image = self.episode_data[step]["env_rgb_array"]
        rl_image = self.episode_data[step].get("work0_rgb_array", None)
        self.set_image(env_image, rl_image)

    def on_loop(self, events: List[pygame.event.Event]):
        if self.get_key(pygame.K_UP) == KeyStatus.PRESSED:
            self.episode += 1
            if self.episode >= self.history.episode_num:
                self.episode = self.history.episode_num - 1
            else:
                self._set_episode()
        elif self.get_key(pygame.K_DOWN) == KeyStatus.PRESSED:
            self.episode -= 1
            if self.episode < 0:
                self.episode = 0
            else:
                self._set_episode()
        elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
            self.step += 1
        elif self.get_key(pygame.K_LEFT) == KeyStatus.PRESSED:
            self.step -= 1
        elif self.get_key(pygame.K_p) == KeyStatus.PRESSED:
            self.env_pause = not self.env_pause
            self.step_t0 = time.time()
        elif self.get_key(pygame.K_r) == KeyStatus.PRESSED:
            self.step = 0
            self.env_pause = True
        elif self.get_key("-") == KeyStatus.PRESSED:
            self.interval *= 2
        elif self.get_key("+") == KeyStatus.PRESSED:
            self.interval /= 2
            if self.interval < 1:
                self.interval = 1

        t = []
        t.append(f"↑↓ : change episode({self.episode})")
        t.append(f"←→ : move step({self.step})")
        if self.env_pause:
            t.append("p  : pause/unpause (Pause)")
        else:
            t.append("p  : pause/unpause (UnPause)")
        t.append("r  : Reset")
        t.append(f"-+ : change speed ({self.interval:.0f}ms; {1000/self.interval:.1f}fps)")
        self.add_hotkey_texts(t)

        if len(self.episode_data) == 0:
            return

        if self.step < 0:
            self.step = 0
            self.env_pause = True
        if self.step >= len(self.episode_data):
            self.step = len(self.episode_data) - 1
            self.env_pause = True

        step_data = self.episode_data[self.step]
        self._set_image(self.step)

        t = [
            "episode      : {}".format(self.episode),
            "total rewards: {}".format(self.episode_info["total_rewards"]),
            "step         : {} / {}".format(step_data["step"], len(self.episode_data) - 1),
            "action       : {}".format(step_data.get("action", None)),
            "next_player_index: {}".format(step_data["next_player_index"]),
            "invalid_actions  : {}".format(step_data["invalid_actions"]),
            "rewards   : {}".format(step_data["rewards"]),
            "done      : {}".format(step_data["done"]),
            "step time : {:.3f}s".format(step_data["time"]),
            "env_info  : {}".format(step_data["env_info"]),
        ]
        if "train_info" in step_data and step_data["train_info"] is not None:
            t.append("train_info: {}".format(step_data["train_info"]))
        if "work0_info" in step_data and step_data["work0_info"] is not None:
            t.append("work0_info: {}".format(step_data["work0_info"]))
        self.add_info_texts(t)

        if not self.env_pause:
            if time.time() - self.step_t0 > self.interval / 1000:
                self.step_t0 = time.time()
                self.step += 1
