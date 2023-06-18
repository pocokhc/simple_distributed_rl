import logging
import time
from typing import Dict, List, Optional, Tuple, Union, cast

import pygame

import srl
from srl.base.define import EnvActionType, KeyBindType, PlayRenderModes
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.spaces.box import BoxSpace
from srl.runner.callback import GameCallback
from srl.runner.game_windows.game_window import GameWindow, KeyStatus

logger = logging.getLogger(__name__)


class PlayableGame(GameWindow):
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
