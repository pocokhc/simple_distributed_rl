import logging
import time
from typing import List, Optional

import pygame

from srl.base.define import PlayRenderModes
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLParameter
from srl.base.rl.worker_run import WorkerRun
from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.runner.core import ProgressOption, play
from srl.runner.game_windows.game_window import GameWindow, KeyStatus

logger = logging.getLogger(__name__)


class _GetRGBCallback(Callback):
    def on_episode_begin(self, info):
        self.steps = []

    def on_step_action_before(self, info) -> None:
        self._tmp_step_env(info)

    def on_step_begin(self, info) -> None:
        self._tmp_step_worker(info)
        self._add_step()

    def on_skip_step(self, info) -> None:
        self._tmp_step_env(info)
        self._add_step(is_skip_step=True)

    def on_episode_end(self, info):
        # 描画用にpolicyを実行
        info["workers"][info["worker_idx"]].policy()

        self._tmp_step_env(info)
        self._tmp_step_worker(info)
        self._add_step(is_skip_step=True)

    # ---------------------------------

    def _tmp_step_env(self, info):
        env: EnvRun = info["env"]
        d = {
            "step": env.step_num,
            "next_player_index": env.next_player_index,
            "state": env.state,
            "invalid_actions": env.get_invalid_actions(),
            "rewards": env.step_rewards,
            "done": env.done,
            "done_reason": env.done_reason,
            "time": info.get("step_time", 0),
            "env_info": env.info,
            "train_time": info.get("train_time", 0),
            "train_info": info.get("train_info", None),
        }
        # --- render info
        d["env_rgb_array"] = env.render_rgb_array()

        self.step_info_env = d

    def _tmp_step_worker(self, info):
        d = {
            "action": info["action"],
        }
        workers: List[WorkerRun] = info["workers"]
        for i, w in enumerate(workers):
            d[f"work{i}_info"] = w.info
            d[f"work{i}_rgb_array"] = w.render_rgb_array()

        self.step_info_worker = d

    def _add_step(self, is_skip_step=False):
        d = {"is_skip_step": is_skip_step}
        d.update(self.step_info_env)
        d.update(self.step_info_worker)
        self.steps.append(d)


class RePlayableGame(GameWindow):
    def __init__(
        self,
        config: Config,
        parameter: Optional[RLParameter] = None,
        # play config
        timeout: int = -1,
        max_steps: int = -1,
        # option
        progress: Optional[ProgressOption] = ProgressOption(),
        # other
        callbacks: List[Callback] = [],
        _is_test: bool = False,  # for test
    ) -> None:
        super().__init__(_is_test=_is_test)

        self.config = config
        self.parameter = parameter
        self.timeout = timeout
        self.max_steps = max_steps
        self.progress = progress
        self.callbacks = callbacks

        env = self.config.make_env()
        self.interval = env.config.render_interval

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
            callbacks = self.callbacks[:]

            history = _GetRGBCallback()
            callbacks.append(history)
            play(
                self.config,
                # stop config
                max_episodes=1,
                timeout=self.timeout,
                max_steps=self.max_steps,
                max_train_count=-1,
                # play config
                train_only=False,
                shuffle_player=False,
                disable_trainer=True,
                enable_profiling=False,
                # play info
                training=False,
                distributed=False,
                render_mode=PlayRenderModes.rgb_array,
                # option
                eval=None,
                progress=self.progress,
                history=None,
                checkpoint=None,
                # other
                callbacks=callbacks,
                parameter=self.parameter,
            )
            self.episode_info = {"total_rewards": 0}
            self.episode_data = history.steps
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
