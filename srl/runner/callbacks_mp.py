import datetime as dt
import json
import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Optional

import numpy as np
import tensorflow as tf
from srl.runner.callbacks import Callback
from srl.utils.common import JsonNumpyEncoder, is_package_installed, listdictdict_to_dictlist, to_str_time

try:
    import psutil
except ImportError:
    pass
ENABLE_PS = is_package_installed("psutil")

try:
    import pynvml
except ImportError:
    pass
ENABLE_NVIDIA = is_package_installed("pynvml")

logger = logging.getLogger(__name__)


class MPCallback(Callback, ABC):
    # all
    def on_init(self, **kwargs) -> None:
        pass  # do nothing

    # main
    def on_start(self, **kwargs) -> None:
        pass  # do nothing

    def on_polling(self, **kwargs) -> None:
        pass  # do nothing

    def on_end(self, **kwargs) -> None:
        pass  # do nothing

    # trainer
    def on_trainer_start(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_train_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_end(self, **kwargs) -> None:
        pass  # do nothing


@dataclass
class MPPrintProgress(MPCallback):

    max_progress_time: int = 60 * 10  # s
    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0
    max_print_worker: int = 5

    def __post_init__(self):
        self.progress_timeout = 5

    def _check_print_progress(self):

        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = _time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_progress_time:
            self.progress_timeout = self.max_progress_time

        return True

    # ---------------------------
    # main
    # ---------------------------
    def on_start(self, config, mp_config, **kwargs):
        self.max_train_count = mp_config.max_train_count
        self.timeout = mp_config.timeout

        print(
            "### env: {}, max train: {}, timeout: {}".format(
                config.env_config.name,
                self.max_train_count,
                to_str_time(self.timeout),
            )
        )

        self.progress_t0 = time.time()

    def on_polling(self, elapsed_time, **kwargs):

        if self._check_print_progress():
            s = dt.datetime.now().strftime("%H:%M:%S")
            s += f" --- {to_str_time(elapsed_time)}"
            if self.timeout > 0:
                remain_time = self.timeout - elapsed_time
                s += f" {to_str_time(remain_time)}(timeout remain time)"
            print(s)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, config, mp_config, **kwargs):
        self.max_train_count = mp_config.max_train_count
        self.progress_t0 = time.time()
        self.progress_history = []

    def on_trainer_end(self, **kwargs):
        self._trainer_print_progress()

    def on_trainer_train_end(
        self,
        remote_memory,
        train_count,
        train_time,
        train_info,
        sync_count,
        valid_reward,
        **kwargs,
    ):
        self.progress_history.append(
            {
                "train_count": train_count,
                "train_time": train_time,
                "train_info": train_info,
                "sync_count": sync_count,
                "memory_len": remote_memory.length(),
                "valid_reward": valid_reward,
            }
        )
        if self._check_print_progress():
            self._trainer_print_progress()

    def _trainer_print_progress(self):
        if len(self.progress_history) == 0:
            return

        info = self.progress_history[-1]
        train_count = info["train_count"]
        sync_count = info["sync_count"]
        memory_len = info["memory_len"]
        train_time = np.mean([t["train_time"] for t in self.progress_history])
        valid_rewards = [t["valid_reward"] for t in self.progress_history if t["valid_reward"] is not None]

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += " trainer :{:8d} tra".format(train_count)
        s += ",{:6.3f}s/tra".format(train_time)
        s += ",{:7d} memory ".format(memory_len)
        s += ",{:6d} sync ".format(sync_count)
        if len(valid_rewards) > 0:
            s += ", {:.4f} val_reward ".format(np.mean(valid_rewards))

        d = listdictdict_to_dictlist(self.progress_history, "train_info")
        for k, arr in d.items():
            s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []

        # --- 残り時間
        if self.max_train_count > 0:
            remain = train_time * (self.max_train_count - train_count)

            s = dt.datetime.now().strftime("%H:%M:%S")
            s += f" --- {train_count} / {self.max_train_count}"
            s += f" {to_str_time(remain)}(train remain time)"
            print(s)

    # ---------------------------
    # worker
    # ---------------------------
    def on_episodes_begin(self, worker_id, **kwargs):
        self.worker_id = worker_id
        if self.worker_id >= self.max_print_worker:
            return

        self.progress_t0 = time.time()
        self.progress_history = []

    def on_episodes_end(self, episode_count, **kwargs):
        if self.worker_id >= self.max_print_worker:
            return
        self._worker_print_progress(episode_count)

    def on_episode_begin(self, **kwargs):
        if self.worker_id >= self.max_print_worker:
            return
        self.history_step = []

    def on_step_end(
        self,
        env,
        workers,
        episode_count,
        step_time,
        **kwargs,
    ):
        if self.worker_id >= self.max_print_worker:
            return
        self.history_step.append(
            {
                "env_info": env.info,
                "work_info": workers[self.print_worker].info,
                "step_time": step_time,
            }
        )
        if self._check_print_progress():
            self._worker_print_progress(episode_count)

    def on_episode_end(
        self,
        episode_step,
        episode_count,
        episode_rewards,
        episode_time,
        worker_indices,
        **kwargs,
    ):
        if self.worker_id >= self.max_print_worker:
            return
        if len(self.history_step) == 0:
            return
        worker_idx = worker_indices[self.print_worker]

        # 1エピソードの結果を平均でまとめる
        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        if "TimeLimit.truncated" in env_info:
            del env_info["TimeLimit.truncated"]
        for k, v in env_info.items():
            env_info[k] = np.mean(v)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        for k, v in work_info.items():
            work_info[k] = np.mean(v)

        epi_data = {
            "episode_count": episode_count,
            "episode_step": episode_step,
            "episode_reward": episode_rewards[worker_idx],
            "episode_time": episode_time,
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "env_info": env_info,
            "work_info": work_info,
        }
        self.progress_history.append(epi_data)

    def _worker_print_progress(self, episode_count):
        if len(self.progress_history) == 0:
            return

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f" worker{self.worker_id:2d}:"

        if len(self.progress_history) == 0:
            if len(self.history_step) > 0:
                step_num = len(self.history_step)
                step_time = np.mean([h["step_time"] for h in self.history_step])
                s += f" {episode_count:8d} episode"
                s += f", {step_num:5d} step"
                s += f", {step_time:.5f}s/step"
        else:
            episode_time = np.mean([h["episode_time"] for h in self.progress_history])
            step_time = np.mean([h["step_time"] for h in self.progress_history])

            s += " {:7d} epi".format(episode_count)
            s += f", {episode_time:.3f}s/epi"

            _r = [h["episode_reward"] for h in self.progress_history]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {min(_r):.3f} {np.mean(_r):.3f} {max(_r):.3f} reward"
            s += f", {np.mean(_s):.1f} step"

            d = listdictdict_to_dictlist(self.progress_history, "env_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"
            d = listdictdict_to_dictlist(self.progress_history, "work_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []


@dataclass
class MPTrainFileLogger(MPCallback):
    dir_path: str = "tmp"

    print_worker: int = 0

    # file logger
    enable_log: bool = True
    log_interval: int = 1  # s

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 10  # s
    test_env_episode = 10

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[TextIOWrapper]] = {}

        self.t0 = time.time()

        # file logger
        if self.enable_log:
            self.log_history = []
            self.log_t0 = self.t0

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self.fp_dict.items():
            if v is not None:
                self.fp_dict[k] = None
                v.close()

    def _check_log_progress(self, time_):
        taken_time = time_ - self.log_t0
        if taken_time < self.log_interval:
            return False
        self.log_t0 = time_
        return True

    def on_init(self, config, mp_config, **kwargs):

        # init
        dir_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{config.env_config.name}_{config.rl_config.getName()}"
        self.base_dir = os.path.join(os.path.abspath(self.dir_path), dir_name)
        logger.debug(f"save path: {self.base_dir}")

        self.param_dir = os.path.join(self.base_dir, "params")
        os.makedirs(self.param_dir, exist_ok=True)

        self.log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # save config
        with open(os.path.join(self.base_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        with open(os.path.join(self.base_dir, "mp_config.json"), "w", encoding="utf-8") as f:
            json.dump(mp_config.to_dict(), f, indent=2)

        # system info
        info = {}
        if ENABLE_PS:
            info["memory size"] = psutil.virtual_memory().total
            info["memory percent"] = psutil.virtual_memory().percent
            info["cpu count"] = len(psutil.Process().cpu_affinity())
            info["cpu(MHz)"] = [c.max for c in psutil.cpu_freq(percpu=True)]
        info["tensorflow device list"] = [d.name for d in tf.config.list_logical_devices()]

        # --- GPU(nvidia) の計測をするかどうか
        global ENABLE_NVIDIA
        if ENABLE_NVIDIA:
            try:
                pynvml.nvmlInit()
                ENABLE_NVIDIA = True
            except Exception:
                ENABLE_NVIDIA = False

        if ENABLE_NVIDIA:
            info["nvidea driver varsion"] = str(pynvml.nvmlSystemGetDriverVersion())
            info["gpu"] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info["gpu"].append(
                    {
                        "device": str(pynvml.nvmlDeviceGetName(handle)),
                        "memory": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                    }
                )
            pynvml.nvmlShutdown()
        with open(os.path.join(self.base_dir, "system.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    def _change_csv(self, filename):
        with open(os.path.join(self.log_dir, filename)) as f:
            for line in f:
                json.load(line)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, config, mp_config, **kwargs):
        self.fp_dict["trainer"] = open(os.path.join(self.log_dir, "trainer.txt"), "w", encoding="utf-8")

        self.max_train_count = mp_config.max_train_count
        self.elapsed_time = 0

        # checkpoint
        if self.enable_checkpoint:
            self.checkpoint_t0 = time.time()
            self.env = config.make_env()

        if ENABLE_NVIDIA:
            pynvml.nvmlInit()

    def on_trainer_end(self, **kwargs):
        if self.enable_log:
            self._trainer_log()
        # if self.enable_checkpoint:
        #    self._save_checkpoint(**kwargs) TODO

        if ENABLE_NVIDIA:
            pynvml.nvmlShutdown()
        self.close()

    def on_trainer_train_end(
        self,
        time_,
        remote_memory,
        train_count,
        train_time,
        train_info,
        sync_count,
        valid_reward,
        **kwargs,
    ):
        self.elapsed_time = time_ - self.t0  # 経過時間
        memory_len = remote_memory.length()

        if self.enable_log:
            self.log_history.append(
                {
                    "train_count": train_count,
                    "train_time": train_time,
                    "train_info": train_info,
                    "sync_count": sync_count,
                    "memory": memory_len,
                    "valid_reward": valid_reward,
                }
            )
            if self._check_log_progress(time_):
                self._trainer_log()

        if self.enable_checkpoint:
            if time_ - self.checkpoint_t0 > self.checkpoint_interval:
                self.checkpoint_t0 = time_
                # self._save_checkpoint(**kwargs) TODO

    def _trainer_print_progress(self):
        if len(self.progress_history) == 0:
            return

        info = self.progress_history[-1]
        train_count = info["train_count"]
        sync_count = info["sync_count"]
        memory_len = info["memory"]
        train_time = np.mean([t["train_time"] for t in self.progress_history])
        valid_rewards = [t["valid_reward"] for t in self.progress_history if t["valid_reward"] is not None]

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += " trainer :{:8d} tra".format(train_count)
        s += ",{:6.3f}s/tra".format(train_time)
        s += ",{:7d} memory ".format(memory_len)
        s += ",{:6d} sync ".format(sync_count)
        if len(valid_rewards) > 0:
            s += ", {:.4f} val_reward ".format(np.mean(valid_rewards))

        d = listdictdict_to_dictlist(self.progress_history, "train_info")
        for k, arr in d.items():
            s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []

        # --- 残り時間
        if self.max_train_count > 0:
            remain = train_time * (self.max_train_count - train_count)

            s = dt.datetime.now().strftime("%H:%M:%S")
            s += f" --- {to_str_time(self.elapsed_time)}(elapsed time)"
            s += f", {to_str_time(remain)}(train remain time)"
            s += f"({train_count} / {self.max_train_count})"
            print(s)

    def _trainer_log(self):
        if self.fp_dict["trainer"] is None:
            return
        if len(self.log_history) == 0:
            return

        info = self.log_history[-1]
        train_count = info["train_count"]
        sync_count = info["sync_count"]
        memory_len = info["memory"]
        train_time = np.mean([t["train_time"] for t in self.log_history])

        # 平均
        info = listdictdict_to_dictlist(self.progress_history, "train_info")
        for k, arr in info.items():
            info[k] = np.mean(arr)

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "rl_memory": memory_len,
            "parameter_sync_count": sync_count,
            "train_count": train_count,
            "train_time": train_time,
            "train_info": info,
        }

        # system info
        if ENABLE_PS:
            d["memory percent"] = psutil.virtual_memory().percent
            d["cpu percent"] = psutil.cpu_percent(percpu=True)
        if ENABLE_NVIDIA:
            d["gpu"] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                d["gpu"].append(
                    {
                        "percent": rate.gpu,
                        "memory percent": rate.memory,
                    }
                )
        self._write_log(self.fp_dict["trainer"], d)

    def _save_checkpoint(
        self,
        train_count,
        parameter,
        **kwargs,
    ):
        # save
        parameter.save(os.path.join(self.param_dir, f"{train_count}.pickle"))

    # ---------------------------
    # worker
    # ---------------------------
    def on_episodes_begin(self, worker_id, **kwargs):
        self.worker_id = worker_id
        self.fp_dict["worker"] = open(os.path.join(self.log_dir, f"worker_{worker_id}.txt"), "w", encoding="utf-8")
        self.progress_history = []

    def on_episodes_end(self, **kwargs):
        if self.enable_log:
            self._worker_log()
        self.close()

    def on_episode_begin(self, **kwargs):
        self.history_step = []

    def on_step_end(
        self,
        env,
        workers,
        episode_count,
        step_time,
        **kwargs,
    ):
        _time = time.time()
        self.elapsed_time = _time - self.t0  # 経過時間

        self.history_step.append(
            {
                "episode_count": episode_count,
                "env_info": env.info,
                "work_info": workers[self.print_worker].info,
                "step_time": step_time,
            }
        )

        if self.enable_log and self._check_log_progress(_time):
            self._worker_log()

    def on_episode_end(
        self,
        episode_step,
        episode_count,
        episode_rewards,
        episode_time,
        worker_indices,
        **kwargs,
    ):
        if len(self.history_step) == 0:
            return
        worker_idx = worker_indices[self.print_worker]

        # 1エピソードの結果を平均でまとめる
        if self.enable_log:
            env_info = listdictdict_to_dictlist(self.history_step, "env_info")
            if "TimeLimit.truncated" in env_info:
                del env_info["TimeLimit.truncated"]
            for k, v in env_info.items():
                env_info[k] = np.mean(v)
            work_info = listdictdict_to_dictlist(self.history_step, "work_info")
            for k, v in work_info.items():
                work_info[k] = np.mean(v)

            epi_data = {
                "episode_count": episode_count,
                "episode_step": episode_step,
                "reward": episode_rewards[worker_idx],
                "episode_time": episode_time,
                "step_time": np.mean([h["step_time"] for h in self.history_step]),
                "env_info": env_info,
                "work_info": work_info,
            }

            self.log_history.append(epi_data)

    def _worker_log(self):
        if self.fp_dict["worker"] is None:
            return
        if len(self.log_history) == 0:
            return

        # 平均
        env_info = listdictdict_to_dictlist(self.log_history, "env_info")
        for k, arr in env_info.items():
            env_info[k] = np.mean(arr)

        # 平均
        work_info = listdictdict_to_dictlist(self.log_history, "work_info")
        for k, arr in work_info.items():
            work_info[k] = np.mean(arr)

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "episode_count": self.log_history[-1]["episode_count"],
            "episode_time": np.mean([h["episode_time"] for h in self.log_history]),
            # "step": np.mean([h["step"] for h in self.log_history]),
            "reward": np.mean([h["reward"] for h in self.log_history]),
            "env_info": env_info,
            "work_info": work_info,
        }
        self._write_log(self.fp_dict["worker"], d)
