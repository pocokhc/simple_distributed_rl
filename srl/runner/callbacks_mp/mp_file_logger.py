import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Optional

import numpy as np
from srl.runner.callback_mp import MPCallback
from srl.runner.file_log_plot import init_file_logger
from srl.utils.common import JsonNumpyEncoder, listdictdict_to_dictlist

try:
    import psutil
except ImportError:
    pass

try:
    import pynvml
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MPFileLogger(MPCallback):
    dir_path: str = "tmp"

    # file logger
    enable_log: bool = True
    log_interval: int = 1  # s

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 20  # s

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[TextIOWrapper]] = {}

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self.fp_dict.items():
            if v is not None:
                self.fp_dict[k] = None
                v.close()

    def on_init(self, config, mp_config, **kwargs):
        self.base_dir, self.log_dir, self.param_dir, self.enable_nvidia, self.enable_ps = init_file_logger(
            config, mp_config, self.dir_path
        )

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, config, mp_config, **kwargs):
        self.fp_dict["trainer"] = open(os.path.join(self.log_dir, "trainer.txt"), "w", encoding="utf-8")

        _time = time.time()

        # file logger
        if self.enable_log:
            self.log_history = []
            self.log_t0 = _time

        # checkpoint
        if self.enable_checkpoint:
            self.checkpoint_t0 = _time
            self.env = config.make_env()

        if self.enable_nvidia:
            pynvml.nvmlInit()

    def on_trainer_end(self, parameter, train_count, **kwargs):
        if self.enable_log:
            self._write_trainer_log()
        if self.enable_checkpoint:
            self._save_parameter(parameter, train_count)

        if self.enable_nvidia:
            pynvml.nvmlShutdown()
        self.close()

    def on_trainer_train_end(
        self,
        remote_memory,
        train_count,
        train_time,
        train_info,
        sync_count,
        valid_reward,
        parameter,
        **kwargs,
    ):
        _time = time.time()

        if self.enable_log:
            self.log_history.append(
                {
                    "train_count": train_count,
                    "train_time": train_time,
                    "train_info": train_info,
                    "sync_count": sync_count,
                    "memory": remote_memory.length(),
                    "valid_reward": valid_reward,
                }
            )
            if _time - self.log_t0 > self.log_interval:
                self.log_t0 = _time
                self._write_trainer_log()

        if self.enable_checkpoint:
            if _time - self.checkpoint_t0 > self.checkpoint_interval:
                self.checkpoint_t0 = _time
                self._save_parameter(parameter, train_count)

    def _write_trainer_log(self):
        if self.fp_dict["trainer"] is None:
            return
        if len(self.log_history) == 0:
            return

        info = self.log_history[-1]
        train_count = info["train_count"]
        train_time = np.mean([t["train_time"] for t in self.log_history])
        sync_count = info["sync_count"]
        memory_len = info["memory"]

        vr_list = [h["valid_reward"] for h in self.log_history if h["valid_reward"] is not None]
        if len(vr_list) == 0:
            valid_reward = None
        else:
            valid_reward = np.mean(vr_list)

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "rl_memory": memory_len,
            "parameter_sync_count": sync_count,
            "train_count": train_count,
            "train_time": train_time,
            "valid_reward": valid_reward,
        }
        info = listdictdict_to_dictlist(self.log_history, "train_info")
        for k, arr in info.items():
            d["train_" + k] = np.mean(arr)

        # system info
        if self.enable_ps:
            d["memory"] = psutil.virtual_memory().percent
            cpus = psutil.cpu_percent(percpu=True)
            for i, cpu in enumerate(cpus):
                d[f"cpu_{i}"] = cpu
            d["cpu_num"] = len(cpus)
        if self.enable_nvidia:
            gpu_num = pynvml.nvmlDeviceGetCount()
            for i in range(gpu_num):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                d[f"gpu_{i}"] = rate.gpu
                d[f"gpu_{i}_memory"] = rate.memory
            d["gpu_num"] = gpu_num

        self._write_log(self.fp_dict["trainer"], d)
        self.log_history = []

    def _save_parameter(
        self,
        parameter,
        train_count,
    ):
        parameter.save(os.path.join(self.param_dir, f"{train_count}.pickle"))

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, actor_id, env, **kwargs):
        self.actor_id = actor_id
        self.fp_dict["actor"] = open(os.path.join(self.log_dir, f"actor_{actor_id}.txt"), "w", encoding="utf-8")

        self.log_history = []
        self.log_t0 = time.time()

        self.player_num = env.player_num

    def on_episodes_end(self, **kwargs):
        if self.enable_log:
            self._write_actor_log()
        self.close()

    def on_episode_begin(self, **kwargs):
        self.history_step = []

    def on_step_end(
        self,
        env,
        workers,
        step_time,
        **kwargs,
    ):
        if not self.enable_log:
            return
        d = {
            "env_info": env.info,
            "step_time": step_time,
        }
        for i, w in enumerate(workers):
            d[f"work_{i}_info"] = w.info
        self.history_step.append(d)

        _time = time.time()
        if _time - self.log_t0 > self.log_interval:
            self.log_t0 = _time
            self._write_actor_log()

    def on_episode_end(
        self,
        episode_count,
        episode_step,
        episode_rewards,
        episode_time,
        worker_indices,
        **kwargs,
    ):
        if len(self.history_step) == 0:
            return

        if not self.enable_log:
            return

        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        if "TimeLimit.truncated" in env_info:
            del env_info["TimeLimit.truncated"]
        for k, v in env_info.items():
            env_info[k] = np.mean(v)

        rewards = [episode_rewards[worker_indices[i]] for i in range(self.player_num)]
        d = {
            "episode_count": episode_count,
            "episode_step": episode_step,
            "episode_rewards": rewards,
            "episode_time": episode_time,
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "env_info": env_info,
        }

        for i in range(self.player_num):
            work_info = listdictdict_to_dictlist(self.history_step, f"work_{i}_info")
            for k, v in work_info.items():
                work_info[k] = np.mean(v)
            d[f"work_{i}_info"] = work_info

        self.log_history.append(d)

    def _write_actor_log(self):
        if self.fp_dict["actor"] is None:
            return
        if len(self.log_history) == 0:
            return

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "episode_count": self.log_history[-1]["episode_count"],
            "episode_step": np.mean([h["episode_step"] for h in self.log_history]),
            "episode_time": np.mean([h["episode_time"] for h in self.log_history]),
            "step_time": np.mean([h["step_time"] for h in self.log_history]),
        }

        env_info = listdictdict_to_dictlist(self.log_history, "env_info")
        for k, arr in env_info.items():
            d["env_" + k] = np.mean(arr)

        for i in range(self.player_num):
            d[f"episode_reward_{i}"] = np.mean([h["episode_rewards"][i] for h in self.log_history])

        for i in range(self.player_num):
            work_info = listdictdict_to_dictlist(self.log_history, f"work_{i}_info")
            for k, arr in work_info.items():
                d[f"work_{i}_" + k] = np.mean(arr)

        self._write_log(self.fp_dict["actor"], d)
        self.log_history = []
