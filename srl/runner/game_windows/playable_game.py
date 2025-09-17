import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pygame

from srl.base.context import RunContext
from srl.base.define import EnvActionType, KeyBindType, RLBaseTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_play_generator import play_generator
from srl.base.spaces.multi import MultiSpace
from srl.runner.game_windows.game_window import GameWindow, KeyStatus

logger = logging.getLogger(__name__)


class _PlayableCallback(RunCallback):
    def __init__(self, env: EnvRun, action_division_num: int):
        self.env = env
        self.valid_actions: list = []
        self.action: int = 0
        self.env.action_space.create_division_tbl(action_division_num)
        self.enc_space = self.env.action_space.set_encode_space(RLBaseTypes.DISCRETE)
        self.action_num = self.enc_space.n

    def on_episode_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        self._read_valid_actions()

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs):
        self._read_valid_actions()

    def _read_valid_actions(self):
        # 入力可能なアクションを読み取り
        invalid_actions = [self.env.action_space.encode_to_space(a) for a in self.env.invalid_actions]
        self.valid_actions = [a for a in range(self.action_num) if a not in invalid_actions]

    def on_step_action_after(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        # アクションでpolicyの結果を置き換える
        manual_action = self.env.action_space.decode_from_space(self.action)
        state.action = manual_action
        state.workers[state.worker_idx].override_action(manual_action)

    def get_env_action(self):
        return self.env.action_space.decode_from_space(self.action)


class PlayableGame(GameWindow):
    def __init__(
        self,
        context: RunContext,
        env: EnvRun,
        worker: WorkerRun,
        trainer: Optional[RLTrainer] = None,
        view_state: bool = True,
        action_division_num: int = 5,
        key_bind: Optional[KeyBindType] = None,
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)
        self.env = env
        self.view_state = view_state
        self.noop = 0
        self.step_time = 0

        self.playable_callback = _PlayableCallback(self.env, action_division_num)

        # --- play ---
        context = context.copy()
        context.callbacks.insert(0, self.playable_callback)
        context.env_render_mode = "rgb_array"
        context.rl_render_mode = "terminal_rgb_array"
        self.gen_play = play_generator(context, env, worker, trainer)
        # 最初まで進める
        while True:
            gen_status, _, gen_state = next(self.gen_play)
            if gen_status == "on_episode_begin":
                assert gen_state.worker is not None
                self.run_worker = gen_state.worker
                break
            logger.debug(f"{gen_status=}")
        logger.debug(f"START {gen_status=}")
        # ---------------------------

        # 初期設定
        self.set_image(self.run_worker.create_render_image())
        self.env_interval = self.env.config.render_interval
        self.scene = "START"
        logger.debug("scene change: START")
        self.mode = "Turn"  # "Turn" or "RealTime"
        self.is_pause = False
        self.cursor_action = 0

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

    def _set_cursor_action(self):
        if self.cursor_action < 0:
            self.cursor_action = 0
        if self.cursor_action >= len(self.playable_callback.valid_actions):
            self.cursor_action = len(self.playable_callback.valid_actions) - 1
        self.playable_callback.action = self.playable_callback.valid_actions[self.cursor_action]

    def on_loop(self, events: List[pygame.event.Event]):
        # --- add hotkey texts
        t = []
        t.append("r  : Reset")
        if self.mode == "RealTime":
            t.append(f"-+ : change speed ({self.env_interval:.0f}ms; {1000 / self.env_interval:.1f}fps)")
            if self.is_pause:
                t.append("p  : pause/unpause (Pause)")
            else:
                t.append("p  : pause/unpause (UnPause)")
            t.append("f  : FrameAdvance")
        self.add_hotkey_texts(t)

        # --- change RealTime option
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
            if self.get_key(pygame.K_UP) == KeyStatus.PRESSED:
                self.mode = "Turn"
            elif self.get_key(pygame.K_DOWN) == KeyStatus.PRESSED:
                self.mode = "RealTime"
            elif self.get_key(pygame.K_RETURN) == KeyStatus.PRESSED:
                self.scene = "RESET"
                logger.debug("scene change: RESET")
            if self.mode == "Turn":
                self.add_info_texts(["> Turn", "  RealTime"])
            else:
                self.add_info_texts(["  Turn", "> RealTime"])
        elif self.scene == "RUNNING":
            self._on_loop_running()
            # sceneが変わっていなければstepを進める
            if self.scene == "RUNNING":
                if self.mode == "Turn":
                    self._on_loop_turn_key(events)
                else:
                    self._on_loop_realtime_key(events)

            # --- add_info_texts 1, key info
            self.add_info_texts([f"Select Action {self.playable_callback.valid_actions}"])
            s = " "
            s1 = str(self.playable_callback.action)
            s2 = self.env.action_to_str(self.playable_callback.get_env_action())
            if s1 == s2:
                s += s1
            else:
                s += f"{s1}({s2})"
            self.add_info_texts([s])

        # --- add_info_texts 2, key bind text
        if self.key_bind is not None:
            t = ["", "- ActionKeys -"]
            if self.mode == "RealTime":
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

        # --- add_info_texts 3, add step info texts
        s = [
            "",
            "- env infos -",
            f"action_space     : {self.env.action_space}",
            f"observation_space: {self.env.observation_space}",
            f"player_num       : {self.env.player_num}",
            f"step   : {self.env.step_num}",
            f"state  : {str(self.env.state)[:30] if self.view_state else 'hidden'}",
            f"next   : {self.env.next_player}",
            f"rewards: {self.env.rewards}",
            f"info   : {self.env.info}",
            f"done   : {self.env.done}({self.env.done_reason})",
            f"time   : {self.step_time * 1000:.1f}ms",
        ]
        self.add_info_texts(s)

        # --- RESETは最後
        if self.scene == "RESET":
            while True:
                gen_status, _, _ = next(self.gen_play)
                if gen_status == "on_step_action_after":
                    break
                logger.debug(f"{gen_status=}")
            logger.debug(f"RESET: {gen_status=}")

            self.env._render.cache_reset()
            self.set_image(self.run_worker.create_render_image())
            self._set_cursor_action()

            self.step_t0 = time.time()
            if self.mode == "Turn":
                self.frameadvance = False
                self.is_pause = True
            else:
                self.frameadvance = False
                self.is_pause = False

            self.scene = "RUNNING"
            logger.debug("scene change: RUNNING")

    def _step(self):
        # --- 1step: アクションの後 on_step_action_after まで進める
        t0 = time.time()
        while True:
            try:
                gen_status, _, state = next(self.gen_play)
            except StopIteration:
                self.pygame_done = True
                break
            if gen_status == "on_episode_end":
                self.scene = "START"
                logger.debug("scene change: START")
                break
            if gen_status == "on_step_action_after":
                # doneの場合はon_episode_endで止めるため止めない
                if not state.env.done:
                    break
        logger.debug(f"step: {gen_status=}")
        self.step_time = time.time() - t0

        self.set_image(self.run_worker.create_render_image())
        self._set_cursor_action()

    def _on_loop_running(self):
        # --- r, reset
        if self.get_key(pygame.K_r) == KeyStatus.PRESSED:
            # on_step_endまで進めてenv done
            while True:
                try:
                    gen_status, _, gen_state = next(self.gen_play)
                    if gen_status == "on_step_end":
                        gen_state.env.abort_episode()
                        break
                except StopIteration:
                    self.pygame_done = True
                    break

            self.scene = "START"
            logger.debug("scene change: START")
            return

        # --- f, RealTime frameadvance
        if (self.mode == "RealTime") and (self.get_key(pygame.K_f) == KeyStatus.PRESSED):
            self.frameadvance = True
            self.is_pause = True
            return

    def _on_loop_turn_key(self, events):
        if self.key_bind is None:
            if self.get_key(pygame.K_LEFT) == KeyStatus.PRESSED:
                self.cursor_action -= 1
                self._set_cursor_action()
            elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
                self.cursor_action += 1
                self._set_cursor_action()

            # key_bindがない場合はアクション決定で1frame進める
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
                self._set_cursor_action()
                self._step()

        else:
            # key bind がある場合は押したら進める
            keys = self.get_pressed_keys()
            if self._get_action_from_key_bind(keys) is not None:
                keys.extend(self.get_down_keys())
                action = self._get_action_from_key_bind(keys)
                if action is None:
                    action = self._get_action_from_key_bind(self.get_pressed_keys())
                if action is not None:
                    self.playable_callback.action = cast(Any, action)
                    self._step()

    def _on_loop_realtime_key(self, events):
        if self.key_bind is None:
            # key bind がない場合は選ぶ
            if self.get_key(pygame.K_LEFT) == KeyStatus.PRESSED:
                self.cursor_action -= 1
                self._set_cursor_action()
            elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
                self.cursor_action += 1
                self._set_cursor_action()
        else:
            # key bind がある場合は押してあるkeyでなければnoop
            action = self._get_action_from_key_bind(self.get_down_keys())
            if action is None:
                action = self.noop
            self.playable_callback.action = cast(Any, action)

        if self.is_pause:
            if self.frameadvance:
                self._step()
                self.frameadvance = False
        elif time.time() - self.step_t0 > self.env_interval / 1000:
            self.step_t0 = time.time()
            self._step()

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
