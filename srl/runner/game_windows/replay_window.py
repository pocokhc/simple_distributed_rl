import logging
import time
from typing import List

import pygame

from srl.base.context import RunContext, RunState
from srl.base.run import core_play
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.runner.game_windows.game_window import GameWindow, KeyStatus

logger = logging.getLogger(__name__)


class _GetRGBCallback(RunCallback):
    def __init__(self, render_player) -> None:
        self.render_player = render_player

    def on_episode_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        self.steps = []
        self.interval = state.env.get_render_interval()

    def on_step_action_after(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._read(context, state)

    def on_skip_step(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._read(context, state, is_skip_step=True)

    def on_episode_end(self, context: RunContext, state: RunStateActor, **kwargs):
        self._read(context, state)

    # ---------------------------------

    def _read(self, context: RunContext, state: RunStateActor, is_skip_step: bool = False):
        env = state.env
        assert env is not None
        # --- env
        d = {
            "step": env.step_num,
            "next_player": env.next_player,
            "state": env.state,
            "invalid_actions": env.get_invalid_actions(),
            "rewards": env.rewards,
            "done": env.done_type.name,
            "env_info": env.info,
            "action": None if env.done else state.action,
            "is_skip_step": is_skip_step,
        }
        # --- worker
        w = state.workers[self.render_player]
        d["work_info"] = w.info

        # --- render
        d["rgb_array"] = state.workers[self.render_player].create_render_image()

        self.steps.append(d)


class RePlayableGame(GameWindow):
    def __init__(
        self,
        context: RunContext,
        state: RunState,
        render_player: int = 0,
        print_state: bool = True,
        callbacks: List[RunCallback] = [],
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)
        self.context = context
        self.state = state
        self.print_state = print_state

        self.history = _GetRGBCallback(render_player)
        self.callbacks = callbacks[:] + [self.history]
        self.interval = -1
        self.episodes_cache = {}
        self.episode = 0
        self._run_episode()

    def _run_episode(self):
        self.env_pause = True
        self.step = 0

        if self.episode in self.episodes_cache:
            cache = self.episodes_cache[self.episode]
            self.episode_info = cache[0]
            self.steps = cache[1]
        else:
            self.context.disable_trainer = True
            self.context.env_render_mode = "rgb_array"
            self.context.rl_render_mode = "terminal_rgb_array"
            core_play.play(self.context, self.state, callbacks=self.callbacks)

            total_rewards = None
            if len(self.history.steps) > 0:
                for h in self.history.steps:
                    if total_rewards is None:
                        total_rewards = h["rewards"]
                    else:
                        total_rewards = [total_rewards[i] + h["rewards"][i] for i in range(len(total_rewards))]
            self.episode_info = {"total_rewards": total_rewards}
            self.steps = self.history.steps
            self.episodes_cache[self.episode] = [
                self.episode_info,
                self.steps,
            ]
            self._set_image(0)

        if self.interval <= 0:
            self.interval = self.history.interval

    def _set_image(self, step: int):
        self.set_image(self.steps[step]["rgb_array"])

    def on_loop(self, events: List[pygame.event.Event]):
        if self.get_key(pygame.K_UP) == KeyStatus.PRESSED:
            self.episode += 1
            self._run_episode()
        elif self.get_key(pygame.K_DOWN) == KeyStatus.PRESSED:
            self.episode -= 1
            if self.episode < 0:
                self.episode = 0
            else:
                self._run_episode()
        elif self.get_key(pygame.K_RIGHT) == KeyStatus.PRESSED:
            self.step += 1
        elif self.get_key(pygame.K_LEFT) == KeyStatus.PRESSED:
            self.step -= 1
            if self.step < 0:
                self.step = 0
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
        t.append(f"-+ : change speed ({self.interval:.0f}ms; {1000 / self.interval:.1f}fps)")
        self.add_hotkey_texts(t)

        if len(self.steps) == 0:
            return

        if self.step < 0:
            self.step = 0
            self.env_pause = True
        if self.step >= len(self.steps):
            self.step = len(self.steps) - 1
            self.env_pause = True

        step_data = self.steps[self.step]
        self._set_image(self.step)

        t = [
            "episode      : {}".format(self.episode),
            "total rewards: {}".format(self.episode_info["total_rewards"]),
            "step         : {} / {}".format(step_data["step"], len(self.steps) - 1),
            "state        : {:20s}".format(str(step_data["state"]) if self.print_state else "hidden"),
            "select action: {}".format(step_data.get("action", None)),
            "next_player  : {}".format(step_data["next_player"]),
            "invalid_acts : {:20s}".format(str(step_data["invalid_actions"])),
            "rewards   : {}".format(step_data["rewards"]),
            "done      : {}".format(step_data["done"]),
            "env_info  : {}".format(step_data["env_info"]),
        ]
        if "train_info" in step_data and step_data["train_info"] is not None:
            t.append("train_info: {}".format(step_data["train_info"]))
        if "work_info" in step_data and step_data["work_info"] is not None:
            t.append("work_info: {}".format(step_data["work_info"]))
        self.add_info_texts(t)

        if not self.env_pause:
            if time.time() - self.step_t0 > self.interval / 1000:
                self.step_t0 = time.time()
                self.step += 1
