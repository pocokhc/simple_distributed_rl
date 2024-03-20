import logging
import time
from typing import Dict, Generator, List, Optional, Tuple, Union, cast

import pygame

from srl.base.define import EnvActionType, KeyBindType
from srl.base.spaces.multi import MultiSpace
from srl.runner.game_windows.game_window import GameWindow, KeyStatus
from srl.runner.runner import CallbackType, Runner

logger = logging.getLogger(__name__)


class PlayableGame(GameWindow):
    def __init__(
        self,
        runner: Runner,
        view_state: bool = True,
        action_division_num: int = 5,
        key_bind: Optional[KeyBindType] = None,
        enable_memory: bool = False,
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)
        self.view_state = view_state

        self.noop = None
        self.step_time = 0
        self.enable_memory = enable_memory

        # --- play ---
        self.env = runner.make_env()
        self.gen_play = cast(
            Generator,
            runner.base_run_play(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                enable_generator=True,
            ),
        )
        gen_status = ""
        while gen_status != "policy":
            self.gen_state, gen_status = next(self.gen_play)
        self.worker = self.gen_state.worker
        # ---------------------------

        # 初期設定
        self.set_image(self.env.render_rgb_array(), None)
        self.env_interval = self.env.config.render_interval
        self.env.action_space.create_division_tbl(action_division_num)

        # --- key bind (扱いやすいように変形) ---
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
        # ----------------------------------------

        self.scene = "START"
        self.mode = "Turn"  # "Turn" or "RealTime"
        self.is_pause = False
        self.action = 0
        self.cursor_action = 0
        self.valid_actions = []

    def _step(self, action):
        t0 = time.time()

        # --- 1step
        self.gen_state, gen_status = self.gen_play.send(action)
        while gen_status != "policy":
            if gen_status == "on_episode_end":
                self.scene = "START"
                break
            try:
                self.gen_state, gen_status = next(self.gen_play)
            except StopIteration:
                self.pygame_done = True
                break

        # --- render
        self.set_image(self.env.render_rgb_array(), self.worker.render_rgb_array())

        # --- action
        self.valid_actions = self.env.get_valid_actions()
        if self.cursor_action >= len(self.valid_actions):
            self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.env.action_space.decode_from_int(self.action)

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
                # on_step_endまで進める
                gen_status = ""
                while gen_status != "on_step_end":
                    try:
                        self.gen_state, gen_status = next(self.gen_play)
                    except StopIteration:
                        break
                assert self.gen_state is not None
                assert self.gen_state.env is not None
                self.gen_state.env.set_done()
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
                        self._step(self.action)
                        self.action = self.valid_actions[self.cursor_action]
                        self.action = self.env.action_space.decode_from_int(self.action)

            elif self.mode == "Turn":
                # key bind があり、Turnの場合は押したら進める
                keys = self.get_pressed_keys()
                if self._get_action_from_key_bind(keys) is not None:
                    keys.extend(self.get_down_keys())
                    self.action = self._get_action_from_key_bind(keys)
                    if self.action is None:
                        self.action = self._get_action_from_key_bind(self.get_pressed_keys())
                    self._step(self.action)

            if self.mode == "RealTime":
                if self.key_bind is not None:
                    # 押してあるkeys
                    action = self._get_action_from_key_bind(self.get_down_keys())
                    if action is not None:
                        self.action = action
                    else:
                        self.action = self.noop
                if self.is_pause:
                    if self.frameadvance:
                        self._step(self.action)
                        self.frameadvance = False
                elif time.time() - self.step_t0 > self.env_interval / 1000:
                    self.step_t0 = time.time()
                    self._step(self.action)

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
            f"observation_space: {self.env.observation_space}",
            f"player_num       : {self.env.player_num}",
            f"step   : {self.env.step_num}",
            f"state  : {str(self.env.state)[:50] if self.view_state else 'hidden'}",
            f"next   : {self.env.next_player_index}",
            f"rewards: {self.env.step_rewards}",
            f"info   : {self.env.info}",
            f"done   : {self.env.done}({self.env.done_reason})",
            f"time   : {self.step_time * 1000:.1f}ms",
        ]
        self.add_info_texts(s)

        # --- RESET は最後
        if self.scene == "RESET":
            self.scene = "RUNNING"
            if self.noop is None:
                self.noop = self.env.action_space.decode_from_int(0)
            self.set_image(self.env.render_rgb_array(), None)
            self.valid_actions = self.env.get_valid_actions()
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

    def _get_action_from_key_bind(self, key):
        key = tuple(sorted(key))
        if isinstance(self.env.action_space, MultiSpace):
            # {match key list : [index, action]}
            self.key_bind = cast(
                Dict[
                    Union[str, int, Tuple[Union[str, int], ...], List[Union[str, int]]],
                    Tuple[int, EnvActionType],
                ],
                self.key_bind,
            )
            acts = self.env.action_space.get_default()
            f = False
            for k in key:
                k = (k,)
                if k in self.key_bind:
                    idx = self.key_bind[k][0]
                    act = self.key_bind[k][1]
                    acts[idx] = act
                    f = True
            if f:
                return acts

        else:
            # {match key list : action}
            self.key_bind = cast(
                Dict[
                    Union[str, int, Tuple[Union[str, int], ...], List[Union[str, int]]],
                    EnvActionType,
                ],
                self.key_bind,
            )
            if key in self.key_bind:
                return self.key_bind[key]
        return None
