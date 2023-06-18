import logging
import time
from typing import List

import pygame

from srl.runner.callbacks.history_episode import HistoryEpisode
from srl.runner.game_windows.game_window import GameWindow, KeyStatus

logger = logging.getLogger(__name__)


class EpisodeReplay(GameWindow):
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
