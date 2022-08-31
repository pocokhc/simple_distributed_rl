import logging
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from srl.runner.callback import Callback
from srl.utils.common import listdictdict_to_dictlist

logger = logging.getLogger(__name__)


@dataclass
class History(Callback):

    target_worker: int = 0
    first_time_clip_num: int = 1

    def on_episodes_begin(self, info):
        self.history = []

    def on_episode_begin(self, info):
        self.history_step = []

    def on_step_end(self, info):
        self.history_step.append(
            {
                "env_info": info["env"].info,
                "work_info": info["workers"][self.target_worker].info,
                "train_info": info["train_info"],
            }
        )

    def on_episode_end(self, info):
        if len(self.history_step) == 0:
            return

        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        for k, v in env_info.items():
            env_info[k] = np.mean(v)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        for k, v in work_info.items():
            work_info[k] = np.mean(v)
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
        else:
            train_info = {}

        player_idx = info["worker_indices"][self.target_worker]
        self.history.append(
            {
                "episode": info["episode_count"],
                "reward": info["episode_rewards"][player_idx],
                "valid_reward": info["valid_reward"],
                "env_info": env_info,
                "work_info": work_info,
                "train_info": train_info,
            }
        )

    # ----------------
    def get_rewards(self) -> List[float]:
        return [h["reward"] for h in self.history]

    def get_valid_rewards(self) -> List[float]:
        return [h["valid_reward"] for h in self.history]

    def plot(self) -> None:
        rewards = [h["reward"] for h in self.history]
        valid_rewards = [h["valid_reward"] for h in self.history]

        rolling_n = int(len(self.history) / 100)

        if len(self.history) > 100:
            alpha = 0.2
            plt.plot(pd.Series(rewards).rolling(rolling_n).mean(), "C0", label=f"reward(mean{rolling_n})")
            plt.plot(
                pd.Series(valid_rewards).rolling(rolling_n).mean(),
                "C1",
                marker=".",
                label=f"valid reward(mean{rolling_n})",
            )
        else:
            alpha = 1
        plt.plot(rewards, "C0", alpha=alpha, label="reward")
        plt.plot(valid_rewards, "C1", alpha=alpha, label="valid reward", marker=".")

        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_info(self, info_name, key) -> None:
        d = listdictdict_to_dictlist(self.history, info_name + "_info")
        rolling_n = int(len(self.history) / 100)

        for target_key, arr in d.items():
            if len(arr) > self.first_time_clip_num:
                arr = arr[self.first_time_clip_num :]
            if key != target_key:
                continue
            if len(self.history) > 100:
                alpha = 0.2
                plt.plot(pd.Series(arr).rolling(rolling_n).mean(), label=f"mean {rolling_n}")
            else:
                alpha = 1
            plt.plot(arr, alpha=alpha, label=key)
            plt.ylabel(key)
            plt.xlabel("episode")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
