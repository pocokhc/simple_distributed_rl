import datetime as dt
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from srl.runner.callback import Callback
from srl.runner.file_log_plot import init_file_logger
from srl.utils.common import JsonNumpyEncoder, listdictdict_to_dictlist

logger = logging.getLogger(__name__)


@dataclass
class FileLogger(Callback):
    dir_path: str = "tmp"

    # file logger
    enable_log: bool = True

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 20  # s

    def __post_init__(self):
        self.fp: Optional[io.TextIOWrapper] = None

    def __del__(self):
        self.close()

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    def on_episodes_begin(self, config, env, **kwargs):
        self.base_dir, self.log_dir, self.param_dir, self.enable_nvidia, self.enable_ps = init_file_logger(
            config, None, self.dir_path
        )
        self.fp = open(os.path.join(self.log_dir, "train.txt"), "w", encoding="utf-8")

        self.log_history = []
        self.log_t0 = time.time()

        self.player_num = env.player_num

    def on_episodes_end(self, **kwargs):
        if self.enable_log:
            self._write_train_log()
        self.close()

    def on_episode_begin(self, **kwargs):
        self.history_step = []

    def on_step_end(self, env, workers, train_info, step_time, train_time, **kwargs):
        if not self.enable_log:
            return
        d = {
            "env_info": env.info,
            "train_info": train_info,
            "step_time": step_time,
            "train_time": train_time,
        }
        for i, w in enumerate(workers):
            d[f"work_{i}_info"] = w.info
        self.history_step.append(d)

        # _time = time.time()
        # if _time - self.log_t0 > self.log_interval:
        #    self.log_t0 = _time
        #    self._write_train_log()

    def on_episode_end(
        self,
        episode_count,
        episode_step,
        episode_rewards,
        episode_time,
        worker_indices,
        valid_reward,
        trainer,
        **kwargs,
    ):
        if len(self.history_step) == 0:
            return

        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        for k, v in env_info.items():
            env_info[k] = np.mean(v)

        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
        else:
            train_info = {}

        rewards = [episode_rewards[worker_indices[i]] for i in range(self.player_num)]
        d = {
            "episode_count": episode_count,
            "episode_step": episode_step,
            "episode_rewards": rewards,
            "episode_time": episode_time,
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "valid_reward": valid_reward,
            "env_info": env_info,
            "train_info": train_info,
        }

        if trainer is None:
            d["train_count"] = 0
        else:
            d["train_count"] = trainer.get_train_count()

        for i in range(self.player_num):
            work_info = listdictdict_to_dictlist(self.history_step, f"work_{i}_info")
            for k, v in work_info.items():
                work_info[k] = np.mean(v)
            d[f"work_{i}_info"] = work_info

        self.log_history.append(d)
        self._write_train_log()

    def _write_train_log(self):
        if self.fp is None:
            return
        if len(self.log_history) == 0:
            return

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "episode_count": self.log_history[-1]["episode_count"],
            "episode_step": np.mean([h["episode_step"] for h in self.log_history]),
            "episode_time": np.mean([h["episode_time"] for h in self.log_history]),
            "step_time": np.mean([h["step_time"] for h in self.log_history]),
            "train_count": self.log_history[-1]["train_count"],
        }

        env_info = listdictdict_to_dictlist(self.log_history, "env_info")
        for k, arr in env_info.items():
            d["env_" + k] = np.mean(arr)

        train_info = listdictdict_to_dictlist(self.log_history, "train_info")
        for k, arr in train_info.items():
            d["train_" + k] = np.mean(arr)

        vr_list = [h["valid_reward"] for h in self.log_history if h["valid_reward"] is not None]
        if len(vr_list) == 0:
            d["valid_reward"] = None
        else:
            d["valid_reward"] = np.mean(vr_list)

        for i in range(self.player_num):
            d[f"episode_reward_{i}"] = np.mean([h["episode_rewards"][i] for h in self.log_history])

        for i in range(self.player_num):
            work_info = listdictdict_to_dictlist(self.log_history, f"work_{i}_info")
            for k, arr in work_info.items():
                d[f"work_{i}_" + k] = np.mean(arr)

        self._write_log(self.fp, d)
        self.log_history = []
