import datetime as dt
import glob
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import srl
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.utils.common import JsonNumpyEncoder, is_package_installed

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())

try:
    import pynvml
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())


@dataclass
class FileLogger(Callback, MPCallback, TrainerCallback):
    tmp_dir: str = "tmp"

    # file logger
    enable_log: bool = True
    log_interval: int = 1  # s

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 20  # s

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[io.TextIOWrapper]] = {}
        self.is_init = False

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self.fp_dict.items():
            if v is not None:
                self.fp_dict[k] = None
                v.close()

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    def on_init(self, info) -> None:
        self._init_file_logger(info["config"], info["mp_config"])

    def _init_file_logger(self, config, mp_config):
        if self.is_init:
            return

        dir_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{config.env_config.name}_{config.rl_config.getName()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        self.base_dir = os.path.join(os.path.abspath(self.tmp_dir), dir_name)
        logger.debug(f"save path: {self.base_dir}")

        self.param_dir = os.path.join(self.base_dir, "params")
        os.makedirs(self.param_dir, exist_ok=True)

        self.log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # ver
        with open(os.path.join(self.base_dir, "version.txt"), "w", encoding="utf-8") as f:
            f.write(srl.__version__)

        # save config
        with open(os.path.join(self.base_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        if mp_config is not None:
            with open(os.path.join(self.base_dir, "mp_config.json"), "w", encoding="utf-8") as f:
                json.dump(mp_config.to_dict(), f, indent=2)

        # system info
        self.enable_ps = is_package_installed("psutil")
        self.enable_nvidia = is_package_installed("pynvml")

        info = {}
        if self.enable_ps:
            info["memory size"] = psutil.virtual_memory().total
            info["memory percent"] = psutil.virtual_memory().percent
            info["cpu count"] = len(psutil.Process().cpu_affinity())
            info["cpu(MHz)"] = [c.max for c in psutil.cpu_freq(percpu=True)]

        # GPU(nvidia) の計測をするかどうか
        if self.enable_nvidia:
            try:
                pynvml.nvmlInit()
                self.enable_nvidia = True
            except Exception:
                self.enable_nvidia = False
                import traceback

                logger.debug(traceback.format_exc())

        if self.enable_nvidia:
            info["nvidia driver version"] = str(pynvml.nvmlSystemGetDriverVersion())
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

        self.is_init = True

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        config = info["config"]
        self._init_file_logger(config, None)
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

    def on_trainer_end(self, info):
        if self.enable_log:
            self._write_trainer_log()
        if self.enable_checkpoint:
            self._save_parameter(info["parameter"], info["train_count"])

        self.close()

    def on_trainer_train(self, info):
        _time = time.time()

        if self.enable_log:
            remote_memory = info["remote_memory"]

            d = {
                "train_count": info["train_count"],
                "train_time": info["train_time"],
                "remote_memory": 0 if remote_memory is None else remote_memory.length(),
            }
            if "sync" in info:
                d["sync"] = info["sync"]

            # --- info は展開して格納
            for k, v in info["train_info"].items():
                if f"train_{k}" in d:
                    k = f"info_{k}"
                d[f"train_{k}"] = v
            self.log_history.append(d)

            if _time - self.log_t0 > self.log_interval:
                self.log_t0 = _time
                self._write_trainer_log()

        if self.enable_checkpoint:
            if _time - self.checkpoint_t0 > self.checkpoint_interval:
                self.checkpoint_t0 = _time
                self._save_parameter(info["parameter"], info["train_count"])

    def _write_trainer_log(self):
        if self.fp_dict["trainer"] is None:
            return
        if len(self.log_history) == 0:
            return

        info = self.log_history[-1]
        train_count = info["train_count"]
        remote_memory = info["remote_memory"]

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "remote_memory": remote_memory,
            "train_count": train_count,
        }
        if "sync" in info:
            d["trainer_parameter_sync_count"] = info["sync"]

        for k in self.log_history[-1].keys():
            if k in [
                "remote_memory",
                "sync",
                "train_count",
            ]:
                continue
            arr = [h[k] for h in self.log_history if (k in h) and (h[k] is not None)]
            n = None if len(arr) == 0 else np.mean(arr)
            if k in d:
                k = f"info_{k}"
            d[k] = n

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
    def on_episodes_begin(self, info):
        self._init_file_logger(info["config"], None)

        self.actor_id = info["actor_id"]
        self.fp_dict["actor"] = open(os.path.join(self.log_dir, f"actor{self.actor_id}.txt"), "w", encoding="utf-8")

        if self.actor_id == 0:
            self.fp_dict["system"] = open(os.path.join(self.log_dir, "system.txt"), "w", encoding="utf-8")
            if self.enable_nvidia:
                pynvml.nvmlInit()
        else:
            self.fp_dict["system"] = None

        self.log_history = []
        self.log_t0 = time.time()

        self.player_num = info["env"].player_num

    def on_episodes_end(self, info):
        if self.enable_log:
            self._write_actor_log()
        if self.actor_id == 0 and self.enable_nvidia:
            pynvml.nvmlShutdown()
        self.close()

    def on_episode_begin(self, info):
        self.history_step = []

    def on_step_end(self, info):
        if not self.enable_log:
            return

        # --- info は展開して格納
        d = {"step_time": info["step_time"]}
        for k, v in info["env"].info.items():
            d[f"env_{k}"] = v
        train_info = info["train_info"]
        if train_info is not None:
            for k, v in train_info.items():
                if k == "time":
                    k = "info_time"
                d[f"train_{k}"] = v
            d["train_time"] = info["train_time"]
        for i, w in enumerate(info["workers"]):
            if w.info is not None:
                for k, v in w.info.items():
                    d[f"work{i}_{k}"] = v
        self.history_step.append(d)

        _time = time.time()
        if _time - self.log_t0 > self.log_interval:
            self.log_t0 = _time
            self._write_actor_log()
            self._write_system_log()

    def on_episode_end(self, info):
        if len(self.history_step) == 0:
            return
        if not self.enable_log:
            return

        remote_memory = info["remote_memory"]
        trainer = info["trainer"]
        episode_rewards = info["episode_rewards"]
        worker_indices = info["worker_indices"]

        d = {
            "episode_count": info["episode_count"],
            "episode_step": info["episode_step"],
            "episode_time": info["episode_time"],
            "eval_reward": info["eval_reward"],
            "remote_memory": 0 if remote_memory is None else remote_memory.length(),
            "train_count": 0 if trainer is None else trainer.get_train_count(),
        }
        if "sync" in info:
            d["worker_parameter_sync_count"] = info["sync"]

        rewards = [episode_rewards[worker_indices[i]] for i in range(self.player_num)]
        for i, r in enumerate(rewards):
            d[f"episode_reward{i}"] = r
        for k in self.history_step[-1].keys():
            arr = [h[k] for h in self.history_step if (k in h) and (h[k] is not None)]
            n = None if len(arr) == 0 else np.mean(arr)
            if k in d:
                k = f"info_{k}"
            d[k] = n

        self.log_history.append(d)

    def _write_actor_log(self):
        if self.fp_dict["actor"] is None:
            return
        if len(self.log_history) == 0:
            return

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "episode_count": self.log_history[-1]["episode_count"],
            "remote_memory": self.log_history[-1]["remote_memory"],
            "train_count": self.log_history[-1]["train_count"],
        }
        for k in self.log_history[-1].keys():
            if k in [
                "episode_count",
                "remote_memory",
                "train_count",
            ]:
                continue
            arr = [h[k] for h in self.log_history if (k in h) and (h[k] is not None)]
            n = None if len(arr) == 0 else np.mean(arr)
            if k in d:
                k = f"info_{k}"
            d[k] = n

        self._write_log(self.fp_dict["actor"], d)
        self.log_history = []

    def _write_system_log(self):
        if self.fp_dict["system"] is None:
            return

        d = {"date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}
        if self.enable_ps:
            d["memory"] = psutil.virtual_memory().percent
            cpus = psutil.cpu_percent(percpu=True)
            for i, cpu in enumerate(cpus):
                d[f"cpu_{i}"] = cpu
            d["cpu"] = np.mean(cpus)
            d["cpu_num"] = len(cpus)

        if self.enable_nvidia:
            gpu_num = pynvml.nvmlDeviceGetCount()
            gpu = 0
            gpu_memory = 0
            for i in range(gpu_num):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                d[f"gpu_{i}"] = rate.gpu
                d[f"gpu_{i}_memory"] = rate.memory
                gpu += rate.gpu
                gpu_memory += rate.memory
            d["gpu"] = gpu / gpu_num
            d["gpu_memory"] = gpu_memory / gpu_num
            d["gpu_num"] = gpu_num

        self._write_log(self.fp_dict["system"], d)


@dataclass
class FileLogPlot:
    def __init__(self) -> None:
        self.logs = []
        self.df = None
        self.distributed = False

    def load(self, path: str):
        if not os.path.isdir(path):
            logger.info(f"Log folder is not found.({path})")
            return
        self.base_dir = path
        self.log_dir = os.path.join(path, "logs")
        self.param_dir = os.path.join(path, "params")
        self.path_mp_config = os.path.join(self.base_dir, "mp_config.json")
        self.path_config = os.path.join(self.base_dir, "config.json")
        self.path_system = os.path.join(self.base_dir, "system.json")
        self.path_version = os.path.join(self.base_dir, "version.txt")
        self.df = None
        self.player_num = 1
        self.actor_num = 1

        # -----------------------------
        # read config
        # ------------------------------
        if os.path.isfile(self.path_version):
            with open(self.path_version) as f:
                v = f.read()
                if v != srl.__version__:
                    logger.warning(f"log version is different({v} != {srl.__version__})")

        if os.path.isfile(self.path_config):
            with open(self.path_config) as f:
                d = json.load(f)
                self.player_num = d["env_config"]["player_num"]

        if os.path.isfile(self.path_mp_config):
            with open(self.path_mp_config) as f:
                d = json.load(f)
                self.actor_num = d["actor_num"]
            self.distributed = True

        # -----------------------------
        # read data
        # ------------------------------
        self.logs = []
        self.logs.extend(self._read_log("trainer.txt", "trainer"))
        for i in range(self.actor_num):
            self.logs.extend(self._read_log(f"actor{i}.txt", f"actor{i}"))
        self.logs.extend(self._read_log("system.txt", "system"))

    def remove_dir(self):
        # --- logs
        for fn in glob.glob(os.path.join(self.log_dir, "*.txt")):
            if os.path.isfile(fn):
                logger.debug(f"remove file: {fn}")
                os.remove(fn)
        logger.debug(f"remove dir : {self.log_dir}")
        os.rmdir(self.log_dir)  # 空のみ対象

        # --- params
        for fn in glob.glob(os.path.join(self.param_dir, "*.*")):
            if os.path.isfile(fn):
                logger.debug(f"remove file: {fn}")
                os.remove(fn)
        logger.debug(f"remove dir : {self.param_dir}")
        os.rmdir(self.param_dir)  # 空のみ対象

        # --- config
        for fn in [
            self.path_config,
            self.path_mp_config,
            self.path_system,
            self.path_version,
        ]:
            if os.path.isfile(fn):
                logger.debug(f"remove file: {fn}")
                os.remove(fn)

        # dir
        logger.debug(f"remove dir : {self.base_dir}")
        os.rmdir(self.base_dir)  # 空のみ対象

    # ----------------------------------------

    def _read_log(self, filename, type_name) -> List[Dict[str, float]]:
        path = os.path.join(self.log_dir, filename)
        if not os.path.isfile(path):
            return []
        data = []
        with open(path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    d["_type"] = type_name
                    data.append(d)
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError {e.args[0]}, '{line.strip()}'")
        return data

    # ----------------------------------------

    def get_logs(self):
        return self.logs

    def get_df(self) -> "pd.DataFrame":
        assert is_package_installed("pandas"), "To use get_df you need to install the 'pandas'. (pip install pandas)"

        if self.df is not None:
            return self.df

        import pandas as pd

        df = pd.DataFrame(self.logs)

        if len(df) == 0:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["date", "episode_count"], inplace=True)
        df["time"] = (df["date"] - df["date"][0]).dt.total_seconds()
        df.set_index("date", inplace=True)

        # nanを補完
        df = df.interpolate(method="time")

        self.df = df
        return self.df

    def plot(
        self,
        plot_left: List[str] = ["episode_reward0", "eval_reward"],
        plot_right: List[str] = [],
        plot_type: str = "",
        aggregation_num: int = 50,
        left_ymin: Optional[float] = None,
        left_ymax: Optional[float] = None,
        right_ymin: Optional[float] = None,
        right_ymax: Optional[float] = None,
    ):
        plot_left = plot_left[:]
        plot_right = plot_right[:]

        if not (is_package_installed("matplotlib") and is_package_installed("pandas")):
            assert (
                False
            ), "To use FileLogPlot you need to install the 'matplotlib', 'pandas'. (pip install matplotlib pandas)"

        assert len(plot_left) > 0

        import matplotlib.pyplot as plt

        df = self.get_df()
        if len(df) == 0:
            return

        if plot_type == "":
            if self.distributed:
                plot_type = "timeline"
            else:
                plot_type = "episode"

        if plot_type == "timeline":
            df = df.drop_duplicates(subset="time")
            x = df["time"]
            xlabel = "time"
        else:
            x = df["episode_count"]
            xlabel = "episode"

        if len(df) > aggregation_num * 2:
            rolling_n = int(len(df) / aggregation_num)
            xlabel = f"{xlabel} ({rolling_n}mean)"
        else:
            rolling_n = 0

        fig, ax1 = plt.subplots()
        color_idx = 0
        for column in plot_left:
            if column not in df:
                continue
            if rolling_n > 0:
                ax1.plot(x, df[column].rolling(rolling_n).mean(), f"C{color_idx}", label=column)
                ax1.plot(x, df[column], f"C{color_idx}", alpha=0.1)
            else:
                ax1.plot(x, df[column], f"C{color_idx}", label=column)
            color_idx += 1
        ax1.legend(loc="upper left")
        if left_ymin is not None:
            ax1.set_ylim(bottom=left_ymin)
        if left_ymax is not None:
            ax1.set_ylim(top=left_ymax)

        if len(plot_right) > 0:
            ax2 = ax1.twinx()
            for column in plot_right:
                if column not in df:
                    continue
                if rolling_n > 0:
                    ax2.plot(x, df[column].rolling(rolling_n).mean(), f"C{color_idx}", label=column)
                    ax2.plot(x, df[column], f"C{color_idx}", alpha=0.1)
                else:
                    ax2.plot(x, df[column], f"C{color_idx}", label=column)
                color_idx += 1
            ax2.legend(loc="upper right")
            if right_ymin is not None:
                ax1.set_ylim(bottom=right_ymin)
            if right_ymax is not None:
                ax1.set_ylim(top=right_ymax)

        ax1.set_xlabel(xlabel)
        plt.grid()
        plt.tight_layout()
        plt.show()
