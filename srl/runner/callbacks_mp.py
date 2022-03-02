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
import psutil
import pynvml
import tensorflow as tf
from srl.runner import sequence
from srl.runner.callbacks import Callback
from srl.utils.common import JsonNumpyEncoder, listdictdict_to_dictlist, to_str_time

G_ENABLE_NVIDIA = False

logger = logging.getLogger(__name__)


class MPCallback(Callback, ABC):
    def on_init(self, info):
        pass

    # main
    def on_start(self, info):
        pass

    def on_polling(self, info):
        pass

    def on_end(self, info):
        pass

    # trainer
    def on_trainer_start(self, info):
        pass

    def on_trainer_train_end(self, info):
        pass

    def on_trainer_end(self, info):
        pass


@dataclass
class TrainFileLogger(MPCallback):
    dir_path: str = "tmp"

    # file logger
    enable_log: bool = True
    log_interval: int = 1  # s

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 10  # s
    test_env_episode = 10

    # print progress
    enable_print_progress: bool = True
    max_progress_time: int = 60 * 10  # s

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[TextIOWrapper]] = {}

        self.t0 = time.time()
        self._step_time = 0
        self.elapsed_time = 0

        # file logger
        if self.enable_log:
            self.log_history = []
            self.log_t0 = self.t0

        # progress
        if self.enable_print_progress:
            self.progress_timeout = 5
            self.progress_t0 = self.t0
            self.progress_history = []

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self.fp_dict.items():
            if v is not None:
                self.fp_dict[k] = None
                v.close()

    def _check_progress(self, is_last):
        if is_last:
            return True
        taken_time = self._step_time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = self._step_time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_progress_time:
            self.progress_timeout = self.max_progress_time

        return True

    def _check_log(self, is_last):
        if is_last:
            return True
        taken_time = self._step_time - self.log_t0
        if taken_time < self.log_interval:
            return False
        self.log_t0 = self._step_time

        return True

    def on_init(self, info):
        config = info["config"]

        # init
        dir_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{config.env_name}_{config.rl_config.getName()}"
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
            json.dump(info["mp_config"].to_dict(), f, indent=2)

        # system info
        info = {
            "memory size": psutil.virtual_memory().total,
            "memory percent": psutil.virtual_memory().percent,
            "cpu count": len(psutil.Process().cpu_affinity()),  # type: ignore
            "cpu(MHz)": [c.max for c in psutil.cpu_freq(percpu=True)],  # type: ignore
        }
        info["tensorflow device list"] = [d.name for d in tf.config.list_logical_devices()]

        # --- GPU(nvidia) の計測をするかどうか
        global G_ENABLE_NVIDIA
        try:
            pynvml.nvmlInit()
            G_ENABLE_NVIDIA = True
        except:
            G_ENABLE_NVIDIA = False

        if G_ENABLE_NVIDIA:
            info["nvidea driver varsion"] = str(pynvml.nvmlSystemGetDriverVersion())
            info["gpu"] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info["gpu"].append(
                    {
                        "device": str(pynvml.nvmlDeviceGetName(handle)),
                        "memory": pynvml.nvmlDeviceGetMemoryInfo(handle).total,  # type: ignore
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
    # main
    # ---------------------------
    def on_start(self, info):
        self.max_train_count = info["mp_config"].max_train_count
        self.timeout = info["mp_config"].timeout
        print(f"### max train: {self.max_train_count}, timeout: {to_str_time(self.timeout)}")

    def on_polling(self, info):
        self._step_time = time.time()
        self.elapsed_time = self._step_time - self.t0  # 経過時間

        if self.timeout > 0:
            if self._check_progress(False):
                remain_time = self.timeout - self.elapsed_time
                s = dt.datetime.now().strftime("%H:%M:%S")
                s += f" --- {to_str_time(self.elapsed_time)}(elapsed time)"
                s += f" {to_str_time(remain_time)}(timeout remain time)"
                print(s)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        self.fp_dict["trainer"] = open(os.path.join(self.log_dir, "trainer.txt"), "w", encoding="utf-8")

        self.max_train_count = info["mp_config"].max_train_count

        # checkpoint
        if self.enable_checkpoint:
            self.checkpoint_t0 = time.time()
            self.env = info["config"].create_env()

        if G_ENABLE_NVIDIA:
            pynvml.nvmlInit()

    def on_trainer_end(self, info):
        self._trainer_print_progress(True)
        self._trainer_log(True)
        self._save_checkpoint(info, True)

        if G_ENABLE_NVIDIA:
            pynvml.nvmlShutdown()
        self.close()

    def on_trainer_train_end(self, info):
        self._step_time = time.time()
        self.elapsed_time = self._step_time - self.t0  # 経過時間

        if self.enable_print_progress:
            self.progress_history.append(info["train_info"])
            self._trainer_print_progress()

        if self.enable_log:
            self.log_history.append(info["train_info"])
            self._trainer_log()

        self._save_checkpoint(info, False)

    def _trainer_print_progress(self, is_last=False):
        if not self.enable_print_progress:
            return
        if len(self.progress_history) == 0:
            return
        if not self._check_progress(is_last):
            return

        info = self.progress_history[-1]
        train_count = info["train"]
        memory_len = info["memory"]
        train_time = np.mean([t["train_time"] for t in self.progress_history])

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += " trainer:{:8d} train".format(train_count)
        s += ",{:6.3f}s/train".format(train_time)
        s += ",{:8d} memory ".format(memory_len)

        d = listdictdict_to_dictlist(self.progress_history, "info")
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

    def _trainer_log(self, is_last: bool = False):
        if not self.enable_log:
            return
        if self.fp_dict["trainer"] is None:
            return
        if len(self.log_history) == 0:
            return
        if not self._check_log(is_last):
            return

        info = self.log_history[-1]
        train_count = info["train"]
        memory_len = info["memory"]
        train_time = np.mean([t["train_time"] for t in self.log_history])

        # 平均
        info = listdictdict_to_dictlist(self.progress_history, "info")
        for k, arr in info.items():
            info[k] = np.mean(arr)

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "train_count": train_count,
            "rl_memory": memory_len,
            "train_time": train_time,
            "train_info": info,
        }

        # system info
        d["memory percent"] = psutil.virtual_memory().percent
        d["cpu percent"] = psutil.cpu_percent(percpu=True)
        if G_ENABLE_NVIDIA:
            d["gpu"] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                d["gpu"].append(
                    {
                        "percent": rate.gpu,  # type: ignore
                        "memory percent": rate.memory,  # type: ignore
                    }
                )
        self._write_log(self.fp_dict["trainer"], d)

    def _save_checkpoint(self, info, is_last):
        if not self.enable_checkpoint:
            return
        if not is_last:
            if self._step_time - self.checkpoint_t0 < self.checkpoint_interval:
                return
        self.checkpoint_t0 = self._step_time

        # parameter
        parameter = info["parameter"]

        # test play
        t0 = time.time()
        config = info["config"].copy()
        config.set_play_config(max_episodes=self.test_env_episode)
        rewards, _, _ = sequence.play(config, parameter, env=self.env)
        reward = np.mean(rewards)

        # save
        train_count = info["train_info"]["train"]
        parameter.save(os.path.join(self.param_dir, f"{train_count}_{reward}.pickle"))

        if self.enable_print_progress:
            print(
                "save params: {} train, test reward {:.3f} {:.3f} {:.3f}, test time {:.3f}s".format(
                    train_count,
                    min(rewards),
                    reward,
                    max(rewards),
                    time.time() - t0,
                )
            )

    # ---------------------------
    # worker
    # ---------------------------
    def on_episodes_begin(self, info) -> None:
        self.worker_id = info["config"].worker_id
        self.fp_dict["worker"] = open(
            os.path.join(self.log_dir, f"worker_{self.worker_id}.txt"), "w", encoding="utf-8"
        )
        self.progress_history = []

    def on_episodes_end(self, info):
        self._worker_print_progress(True)
        self._worker_log(True)
        self.close()

    def on_episode_begin(self, info) -> None:
        self.history_step = []

    def on_step_end(self, info) -> None:
        self._step_time = time.time()
        self.elapsed_time = self._step_time - self.t0  # 経過時間

        self.history_step.append(
            {
                "episode_count": info["episode_count"],
                "step": info["step"],
                "env_info": info["env_info"],
                "work_info": info["work_info"],
                "train_info": info["train_info"],
            }
        )

        self._worker_print_progress(False)
        self._worker_log(False)

    def on_episode_end(self, info) -> None:
        # step情報をまとめる
        if self.enable_print_progress or self.enable_log:
            env_info = listdictdict_to_dictlist(self.history_step, "env_info")
            if "TimeLimit.truncated" in env_info:
                del env_info["TimeLimit.truncated"]
            for k, v in env_info.items():
                env_info[k] = np.mean(v)
            work_info = listdictdict_to_dictlist(self.history_step, "work_info")
            for k, v in work_info.items():
                work_info[k] = np.mean(v)
        else:
            env_info = None
            work_info = None

        if self.enable_print_progress:
            self.progress_history.append(
                {
                    "episode": info["episode_count"],
                    "episode_time": info["episode_time"],
                    "step": info["step"],
                    "reward": info["reward"],
                    "env_info": env_info,
                    "work_info": work_info,
                }
            )
        if self.enable_log:
            self.log_history.append(
                {
                    "episode": info["episode_count"],
                    "episode_time": info["episode_time"],
                    "step": info["step"],
                    "reward": info["reward"],
                    "env_info": env_info,
                    "work_info": work_info,
                }
            )

    def _worker_print_progress(self, is_last):
        if not self.enable_print_progress:
            return
        if not self._check_progress(is_last):
            return

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f"{self.worker_id:7d}w:"

        if len(self.progress_history) == 0:
            if len(self.history_step) > 0:
                info = self.history_step[-1]
                s += "{:8d} episode, {:8d} step".format(info["episode_count"], info["step"])
        else:
            s += "{:8d} epi".format(self.progress_history[-1]["episode"])

            _t = np.mean([h["episode_time"] for h in self.progress_history])
            s += ", {:.2f}s/epi".format(_t)

            _r = [h["reward"] for h in self.progress_history]
            _s = [h["step"] for h in self.progress_history]
            s += f", {min(_r):.3f} {np.mean(_r):.3f} {max(_r):.3f} reward"
            s += f", {np.mean(_s):.1f} step "

            d = listdictdict_to_dictlist(self.progress_history, "env_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"
            d = listdictdict_to_dictlist(self.progress_history, "work_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []

    def _worker_log(self, is_last):
        if not self.enable_log:
            return
        if self.fp_dict["worker"] is None:
            return
        if len(self.log_history) == 0:
            return
        if not self._check_log(is_last):
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
            "episode": self.log_history[-1]["episode"],
            "episode_time": np.mean([h["episode_time"] for h in self.log_history]),
            "step": np.mean([h["step"] for h in self.log_history]),
            "reward": np.mean([h["reward"] for h in self.log_history]),
            "env_info": env_info,
            "work_info": work_info,
        }
        self._write_log(self.fp_dict["worker"], d)


if __name__ == "__main__":
    pass
