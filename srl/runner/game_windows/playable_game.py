import logging
import random
import time
from typing import Dict, List, Optional, Tuple, cast

import pygame

from srl.base.define import EnvActionType, KeyBindType, RenderModes
from srl.base.run.core import RunStateActor
from srl.runner.callback import GameCallback
from srl.runner.game_windows.game_window import GameWindow, KeyStatus
from srl.runner.runner import Runner
from srl.utils import common

logger = logging.getLogger(__name__)


class PlayableGame(GameWindow):
    def __init__(
        self,
        runner: Runner,
        key_bind: KeyBindType = None,
        enable_memory: bool = False,
        callbacks: List[GameCallback] = [],
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)

        self.runner = runner
        self.callbacks = callbacks[:]

        self.noop = None
        self.step_time = 0
        self.enable_memory = enable_memory

        # --- env/workers/trainer ---
        self.state = RunStateActor(
            runner.make_env(is_init=True),
            runner.make_workers(),
            runner.make_memory(),
            runner.make_parameter(),
            trainer=None,
        )
        # ---------------------------

        # --- key bind (扱いやすいように変形) ---
        if key_bind is None:
            key_bind = self.state.env.get_key_bind()
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

        # --- random seed ---
        if runner.config.seed is not None:
            common.set_seed(runner.config.seed, runner.config.seed_enable_gpu)
            self.state.episode_seed = random.randint(0, 2**16)
        # -------------------

        # --- reset ---
        self.state.env.reset(render_mode=RenderModes.rgb_array, seed=self.state.episode_seed)
        if self.state.episode_seed is not None:
            self.state.episode_seed += 1
        self.env_interval = self.state.env.config.render_interval
        self.set_image(self.state.env.render_rgb_array(), None)
        if self.enable_memory:
            self.state.worker_idx = self.state.env.next_player_index
            [
                w.on_reset(i, training=self.runner.context.training, render_mode=self.runner.context.render_mode)
                for i, w in enumerate(self.state.workers)
            ]
        else:
            self.remote_memory = None
        # -------------

        self.scene = "START"
        self.mode = "Turn"  # "Turn" or "RealTime"
        self.is_pause = False
        self.action = 0
        self.cursor_action = 0
        self.valid_actions = []

        [c.on_game_init(self.runner) for c in self.callbacks]

    def _env_step(self, action):
        assert self.state.env is not None
        t0 = time.time()

        # --- worker.policy
        if self.enable_memory:
            _ = self.state.workers[self.state.worker_idx].policy()

        # --- env.step
        self.state.action = action
        if self.runner.env_config.frameskip == 0:
            self.state.env.step(action)
        else:

            def __f():
                [c.on_skip_step(self.runner) for c in self.callbacks]

            self.state.env.step(action, __f)
        worker_idx = self.state.env.next_player_index

        # --- worker.on_step
        if self.enable_memory:
            [w.on_step() for w in self.state.workers]

        # callbacks
        [c.on_game_step_end(self.runner) for c in self.callbacks]
        self.state.worker_idx = worker_idx

        # --- render ---
        self.set_image(self.state.env.render_rgb_array(), None)
        invalid_actions = self.state.env.get_invalid_actions()
        self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
        if self.cursor_action >= len(self.valid_actions):
            self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.state.env.action_space.decode_from_int(self.action)
        # ---------------

        # --- done
        if self.state.env.done:
            self.scene = "START"
            [c.on_game_end(self.runner) for c in self.callbacks]

        self.step_time = time.time() - t0

    def on_loop(self, events: List[pygame.event.Event]):
        assert self.state.env is not None

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
                    self.action = self.state.env.action_space.decode_from_int(self.action)
                elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
                    self.cursor_action += 1
                    if self.cursor_action >= len(self.valid_actions):
                        self.cursor_action = len(self.valid_actions) - 1
                    self.action = self.valid_actions[self.cursor_action]
                    self.action = self.state.env.action_space.decode_from_int(self.action)

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
                        self.action = self.state.env.action_space.decode_from_int(self.action)

            elif self.mode == "Turn":
                # key bind があり、Turnの場合は押したら進める
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
            s2 = self.state.env.action_to_str(self.action)
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
                s2 = self.state.env.action_to_str(val)
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
            f"action_space     : {self.state.env.action_space}",
            f"observation_type : {self.state.env.observation_type.name}",
            f"observation_space: {self.state.env.observation_space}",
            f"player_num       : {self.state.env.player_num}",
            f"step   : {self.state.env.step_num}",
            f"next   : {self.state.env.next_player_index}",
            f"rewards: {self.state.env.step_rewards}",
            f"info   : {self.state.env.info}",
            f"done   : {self.state.env.done}({self.state.env.done_reason})",
            f"time   : {self.step_time*1000:.1f}ms",
        ]
        self.add_info_texts(s)

        # --- RESET は最後
        if self.scene == "RESET":
            self.scene = "RUNNING"
            # if isinstance(self.state.env.action_space, BoxSpace):
            #    self.state.env.action_space.create_division_tbl(self.action_division_num)
            if self.noop is None:
                self.noop = self.state.env.action_space.decode_from_int(0)
            self.state.env.reset()
            self.set_image(self.state.env.render_rgb_array(), None)
            self.action_size = self.state.env.action_space.n
            invalid_actions = self.state.env.get_invalid_actions()
            self.valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
            if self.cursor_action >= len(self.valid_actions):
                self.cursor_action = len(self.valid_actions) - 1
            self.action = self.valid_actions[self.cursor_action]
            self.action = self.state.env.action_space.decode_from_int(self.action)

            self.step_t0 = time.time()
            if self.mode == "Turn":
                self.frameadvance = False
                self.is_pause = True
            else:
                self.frameadvance = False
                self.is_pause = False

            [c.on_game_begin(self.runner) for c in self.callbacks]
