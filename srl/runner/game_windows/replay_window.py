import logging
import time
from typing import List

import pygame

from srl.base.run.callback import RunCallback
from srl.base.run.context import RunContext
from srl.base.run.core import RunState
from srl.runner.game_windows.game_window import GameWindow, KeyStatus
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


class _GetRGBCallback(RunCallback):
    def on_episode_begin(self, context: RunContext, state: RunState):
        self.steps = []

    def on_step_action_before(self, context: RunContext, state: RunState) -> None:
        self._tmp_step_env(context, state)

    def on_step_begin(self, context: RunContext, state: RunState) -> None:
        self._tmp_step_worker(context, state)
        self._add_step()

    def on_skip_step(self, context: RunContext, state: RunState) -> None:
        self._tmp_step_env(context, state)
        self._add_step(is_skip_step=True)

    def on_episode_end(self, context: RunContext, state: RunState):
        self._tmp_step_env(context, state)
        self._tmp_step_worker(context, state)
        self._add_step(is_skip_step=True)

    # ---------------------------------

    def _tmp_step_env(self, context: RunContext, state: RunState):
        env = state.env
        assert env is not None
        d = {
            "step": env.step_num,
            "next_player_index": env.next_player_index,
            "state": env.state,
            "invalid_actions": env.get_invalid_actions(),
            "rewards": env.step_rewards,
            "done": env.done,
            "done_reason": env.done_reason,
            "env_info": env.info,
        }
        # --- render info
        d["env_rgb_array"] = env.render_rgb_array()

        self.step_info_env: dict = d

    def _tmp_step_worker(self, context: RunContext, state: RunState):
        d: dict = {
            "action": state.action,
        }
        for i, w in enumerate(state.workers):
            d[f"work{i}_info"] = w.info
            d[f"work{i}_rgb_array"] = w.render_rgb_array()

        self.step_info_worker: dict = d

    def _add_step(self, is_skip_step=False):
        d = {"is_skip_step": is_skip_step}
        d.update(self.step_info_env)
        d.update(self.step_info_worker)
        self.steps.append(d)


class RePlayableGame(GameWindow):
    def __init__(
        self,
        runner: Runner,
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)
        self.runner = runner

        self.history = _GetRGBCallback()
        self.interval = self.runner.env.config.render_interval
        self.episodes_cache = {}
        self.episode = 0
        self._init_episode()

    def _init_episode(self):
        self.env_pause = True
        self.step = 0

        if self.episode in self.episodes_cache:
            cache = self.episodes_cache[self.episode]
            self.episode_info = cache[0]
            self.episode_data = cache[1]
        else:
            self.runner.context.disable_trainer = True
            self.runner.base_run_play(
                trainer_only=False,
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                callbacks=[self.history],
            )

            self.episode_info = {"total_rewards": 0}
            self.episode_data = self.history.steps
            self.episodes_cache[self.episode] = [
                self.episode_info,
                self.episode_data,
            ]
            self._set_image(0)

    def _set_image(self, step: int):
        env_image = self.episode_data[step]["env_rgb_array"]
        rl_image = self.episode_data[step].get("work0_rgb_array", None)
        self.set_image(env_image, rl_image)

    def on_loop(self, events: List[pygame.event.Event]):
        if self.get_key(pygame.K_UP) == KeyStatus.PRESSED:
            self.episode += 1
            self._init_episode()
        elif self.get_key(pygame.K_DOWN) == KeyStatus.PRESSED:
            self.episode -= 1
            if self.episode < 0:
                self.episode = 0
            else:
                self._init_episode()
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
            # "total rewards: {}".format(self.episode_info["total_rewards"]),
            "step         : {} / {}".format(step_data["step"], len(self.episode_data) - 1),
            "action       : {}".format(step_data.get("action", None)),
            "next_player_index: {}".format(step_data["next_player_index"]),
            "invalid_actions  : {}".format(step_data["invalid_actions"]),
            "rewards   : {}".format(step_data["rewards"]),
            "done      : {}".format(step_data["done"]),
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
