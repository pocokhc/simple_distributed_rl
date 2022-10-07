import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pygame
import srl
from srl.base.define import EnvAction, KeyBindType, PlayRenderMode
from srl.base.env.config import EnvConfig
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.config import RLConfig
from srl.runner.callback import GameCallback
from srl.runner.callbacks.file_log_reader import FileLogReader
from srl.utils import pygame_wrapper as pw

logger = logging.getLogger(__name__)


class _GameWindow(ABC):
    def __init__(self) -> None:
        self.title: str = ""
        self.padding: int = 4
        self.img_dir = os.path.join(os.path.dirname(__file__), "img")
        self.pressed_keys = []
        self.relevant_keys = []

        self.org_env_w = 0
        self.org_env_h = 0
        self.rl_w = 0
        self.rl_h = 0
        self.info_w = 100
        self.info_h = 100
        self.scale = 1
        self.resize(1.0)

    @abstractmethod
    def on_keydown(self, event):
        raise NotImplementedError()

    @abstractmethod
    def on_loop(self):
        raise NotImplementedError()

    def play(self):

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

            is_update = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame_done = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.relevant_keys and event.key in self.pressed_keys:
                        self.pressed_keys.remove(event.key)
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.relevant_keys and event.key not in self.pressed_keys:
                        self.pressed_keys.append(event.key)

                    if event.key == pygame.K_ESCAPE:
                        pygame_done = True
                    elif event.unicode == "1":
                        self.scale = 0.5
                        is_update = True
                    elif event.unicode == "2":
                        self.scale = 1.0
                        is_update = True
                    elif event.unicode == "3":
                        self.scale = 1.5
                        is_update = True
                    elif event.unicode == "4":
                        self.scale = 2.0
                        is_update = True
                    elif event.unicode == "5":
                        self.scale = 3.0
                        is_update = True
                    elif event.unicode == "6":
                        self.scale = 4.0
                        is_update = True

                    self.on_keydown(event)

            # --- window check
            if self.org_env_w < self.env_image.shape[1]:
                self.org_env_w = self.env_image.shape[1]
                is_update = True
            if self.org_env_h < self.env_image.shape[0]:
                self.org_env_h = self.env_image.shape[0]
                is_update = True
            if self.rl_w < self.rl_image.shape[1]:
                self.rl_w = self.rl_image.shape[1]
                is_update = True
            if self.rl_h < self.rl_image.shape[0]:
                self.rl_h = self.rl_image.shape[0]
                is_update = True

            self.screen.fill((0, 0, 0))

            # --- image
            pw.draw_image_rgb_array(
                self.screen,
                0,
                0,
                self.env_image,
                resize=(self.env_image.shape[1] * self.scale, self.env_image.shape[0] * self.scale),
            )
            pw.draw_image_rgb_array(self.screen, self.base_rl_x, self.base_rl_y, self.rl_image)

            # --- info
            self.info_texts = []
            self.hotkey_texts = [
                "- Hotkeys -",
                f"1-6: change screen size (x{self.scale:.1f})",
            ]

            self.on_loop()
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
                is_update = True
            if self.info_h < height:
                self.info_h = height
                is_update = True

            pygame.display.flip()
            clock.tick(60)

            if is_update:
                self.resize(self.scale)

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
        config: EnvConfig,
        players: List[Union[None, str, RLConfig]] = [None],
        key_bind: KeyBindType = None,
        action_division_num: int = 5,
        callbacks: List[GameCallback] = [],
    ) -> None:
        super().__init__()
        self.config = config
        self.env = srl.make_env(config)
        self.action_division_num = action_division_num
        self.players = players  # TODO
        self.callbacks = callbacks[:]
        self.noop = None

        # 扱いやすいように変形
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

        self.key_bind = cast(Dict[Optional[Tuple[int]], EnvAction], self.key_bind)
        self.relevant_keys = []
        if self.key_bind is not None:
            for keys in self.key_bind.keys():
                self.relevant_keys.extend(keys)
            self.relevant_keys = set(self.relevant_keys)

        self.env.set_render_mode(PlayRenderMode.rgb_array)
        self.env.reset()
        env_image = self.env.render_rgb_array()
        self.env_interval = self.env.render_interval
        self.set_image(env_image, None)

        self.scene = "START"
        self.mode = "Turn"  # "Turn" or "RealTime"
        self.is_pause = False
        self.action = 0
        self.cursor_action = 0
        self.valid_actions = []

        self._callback_info = {
            "env_config": config,
            "env": self.env,
        }
        [c.on_game_init(self._callback_info) for c in self.callbacks]

    def on_keydown(self, event):
        if self.scene == "START":
            if event.key == pygame.K_UP:
                self.mode = "Turn"
            elif event.key == pygame.K_DOWN:
                self.mode = "RealTime"
            elif event.key == pygame.K_RETURN:
                self.scene = "RESET"
        elif self.scene == "RUNNING":
            if self.key_bind is None:
                if event.key == pygame.K_LEFT:
                    self.cursor_action -= 1
                    if self.cursor_action < 0:
                        self.cursor_action = 0
                    self.action = self.valid_actions[self.cursor_action]
                    self.action = self.env.action_space.action_discrete_decode(self.action)
                elif event.key == pygame.K_RIGHT:
                    self.cursor_action += 1
                    if self.cursor_action >= len(self.valid_actions):
                        self.cursor_action = len(self.valid_actions) - 1
                    self.action = self.valid_actions[self.cursor_action]
                    self.action = self.env.action_space.action_discrete_decode(self.action)
            else:
                # keybindがあり、turnの場合は押したら進める
                if self.mode == "Turn":
                    action = self._get_keybind_action()
                    if action is not None:
                        self.action = action
                        self._env_step(self.action)

            if event.key == pygame.K_r:
                self.scene = "START"
            elif self.mode == "Turn":
                if self.key_bind is None:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_z:
                        self._env_step(self.action)
                        self.action = self.valid_actions[self.cursor_action]
                        self.action = self.env.action_space.action_discrete_decode(self.action)

            elif self.mode == "RealTime":
                if event.key == pygame.K_f:
                    self.frameadvance = True
                    self.is_pause = True

        if self.mode == "RealTime":
            if event.unicode == "-":
                self.env_interval *= 2
            elif event.unicode == "+":
                self.env_interval /= 2
                if self.env_interval < 1:
                    self.env_interval = 1
            elif event.key == pygame.K_p:
                self.is_pause = not self.is_pause

    def _get_keybind_action(self):
        assert self.key_bind is not None
        key = tuple(sorted(self.pressed_keys))
        if key in self.key_bind:
            return self.key_bind[key]
        return self.noop

    def _env_step(self, action):
        self.env.step(action)
        self.set_image(self.env.render_rgb_array(), None)
        invalid_actions = self.env.get_invalid_actions()
        self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
        if self.cursor_action >= len(self.valid_actions):
            self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.env.action_space.action_discrete_decode(self.action)

        self._callback_info["action"] = action
        [c.on_game_step_end(self._callback_info) for c in self.callbacks]

        if self.env.done:
            self.scene = "START"
            [c.on_game_end(self._callback_info) for c in self.callbacks]

    def on_loop(self):
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

        if self.scene == "START":
            if self.mode == "Turn":
                self.add_info_texts(["> Turn", "  RealTime"])
            else:
                self.add_info_texts(["  Turn", "> RealTime"])
        elif self.scene == "RUNNING":
            self.add_info_texts([f"Select Action {self.valid_actions}"])

            s = " "
            s1 = str(self.action)
            s2 = self.env.action_to_str(self.action)
            if s1 == s2:
                s += s1
            else:
                s += f"{s1}({s2})"
            self.add_info_texts([s])

            if self.mode == "RealTime":
                # none じゃない場合は入力してるキーをアクションにする
                if self.key_bind is not None:
                    self.action = self._get_keybind_action()
                if self.is_pause:
                    if self.frameadvance:
                        self._env_step(self.action)
                        self.action = self.noop
                        self.frameadvance = False
                elif time.time() - self.step_t0 > self.env_interval / 1000:
                    self.step_t0 = time.time()
                    self._env_step(self.action)
                    self.action = self.noop

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
        ]
        self.add_info_texts(s)

        if self.scene == "RESET":
            self.scene = "RUNNING"
            if isinstance(self.env.action_space, BoxSpace):
                self.env.action_space.set_action_division(self.action_division_num)
            if self.noop is None:
                self.noop = self.env.action_space.action_discrete_decode(0)
            self.env.reset()
            self.set_image(self.env.render_rgb_array(), None)
            self.action_size = self.env.action_space.get_action_discrete_info()
            invalid_actions = self.env.get_invalid_actions()
            self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
            if self.cursor_action >= len(self.valid_actions):
                self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.env.action_space.action_discrete_decode(self.action)

            self.step_t0 = time.time()
            self.frameadvance = False

            [c.on_game_begin(self._callback_info) for c in self.callbacks]


class EpisodeReplay(_GameWindow):
    def __init__(self, history: FileLogReader) -> None:
        super().__init__()

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

    def on_keydown(self, event):
        if event.key == pygame.K_UP:
            self.episode += 1
            if self.episode >= self.history.episode_num:
                self.episode = self.history.episode_num - 1
            else:
                self._set_episode()
        elif event.key == pygame.K_DOWN:
            self.episode -= 1
            if self.episode < 0:
                self.episode = 0
            else:
                self._set_episode()
        elif event.key == pygame.K_RIGHT:
            self.step += 1
        elif event.key == pygame.K_LEFT:
            self.step -= 1
        elif event.key == pygame.K_p:
            self.env_pause = not self.env_pause
            self.step_t0 = time.time()
        elif event.key == pygame.K_r:
            self.step = 0
            self.env_pause = True
        elif event.unicode == "-":
            self.interval *= 2
        elif event.unicode == "+":
            self.interval /= 2
            if self.interval < 1:
                self.interval = 1

    def on_loop(self):
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
