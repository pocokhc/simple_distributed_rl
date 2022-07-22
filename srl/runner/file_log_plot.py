import datetime as dt
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import srl
import tensorflow as tf
from srl.utils.common import is_package_installed

try:
    import psutil
except ImportError:
    pass

try:
    import pynvml
except ImportError:
    pass


logger = logging.getLogger(__name__)


def init_file_logger(config: "srl.runner.callback.Config", mp_config: Optional["srl.runner.mp.Config"], path: str):

    dir_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name += f"_{config.env_config.name}_{config.rl_config.getName()}"
    base_dir = os.path.join(os.path.abspath(path), dir_name)
    logger.debug(f"save path: {base_dir}")

    param_dir = os.path.join(base_dir, "params")
    os.makedirs(param_dir, exist_ok=True)

    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ver
    with open(os.path.join(base_dir, "version.txt"), "w", encoding="utf-8") as f:
        f.write(srl.__version__)

    # save config
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    if mp_config is not None:
        with open(os.path.join(base_dir, "mp_config.json"), "w", encoding="utf-8") as f:
            json.dump(mp_config.to_dict(), f, indent=2)

    # system info
    enable_ps = is_package_installed("psutil")
    enable_nvidia = is_package_installed("pynvml")

    info = {}
    if enable_ps:
        info["memory size"] = psutil.virtual_memory().total
        info["memory percent"] = psutil.virtual_memory().percent
        info["cpu count"] = len(psutil.Process().cpu_affinity())
        info["cpu(MHz)"] = [c.max for c in psutil.cpu_freq(percpu=True)]
    info["tensorflow device list"] = [d.name for d in tf.config.list_logical_devices()]

    # GPU(nvidia) の計測をするかどうか
    if enable_nvidia:
        try:
            pynvml.nvmlInit()
            enable_nvidia = True
        except Exception:
            enable_nvidia = False

    if enable_nvidia:
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
    with open(os.path.join(base_dir, "system.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return base_dir, log_dir, param_dir, enable_nvidia, enable_ps


@dataclass
class FileLogPlot:
    def __init__(self) -> None:
        self._is_load = False

    def load(self, path: str, remove_dir: bool = False):
        assert os.path.isdir(path)
        self.base_dir = path
        self.log_dir = os.path.join(path, "logs")
        self.param_dir = os.path.join(path, "params")
        self.path_mp_config = os.path.join(self.base_dir, "mp_config.json")
        self.path_config = os.path.join(self.base_dir, "config.json")
        self.path_system = os.path.join(self.base_dir, "system.json")
        self.path_version = os.path.join(self.base_dir, "version.txt")
        self.df = None

        # -----------------------------
        # read config
        # ------------------------------
        self.is_mp = os.path.isfile(self.path_mp_config)

        with open(self.path_version) as f:
            v = f.read()
            if v != srl.__version__:
                logger.warning(f"log version is different({v} != {srl.__version__})")

        with open(self.path_config) as f:
            d = json.load(f)
            self.player_num = d["env_config"]["player_num"]

        if self.is_mp:
            with open(self.path_mp_config) as f:
                d = json.load(f)
                self.actor_num = d["actor_num"]

        # -----------------------------
        # read df
        # ------------------------------
        self._read_df()
        self._is_load = True

        # -----------------------------
        # remove dir
        # ------------------------------
        if remove_dir:
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

    def _read_log(self, filename):
        data = []
        with open(os.path.join(self.log_dir, filename)) as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError {e.args[0]}, '{line.strip()}'")
        return pd.DataFrame(data)

    def _read_df(self, actor_id: int = 0):
        if self.df is not None:
            return

        if self.is_mp:
            df_trainer = self._read_log("trainer.txt")
            df_actor = self._read_log(f"actor_{actor_id}.txt")
            self.df = pd.merge(df_trainer, df_actor, on="date", how="outer")

        else:
            self.df = self._read_log("train.txt")

        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.sort_values(["date", "episode_count"], inplace=True)
        self.df["time"] = (self.df["date"] - self.df["date"][0]).dt.total_seconds()
        self.df.set_index("date", inplace=True)

        # nanを補完
        self.df = self.df.interpolate(method="time")

    # ----------------------------------------

    def get_df(self) -> pd.DataFrame:
        return self.df

    def plot(
        self,
        plot_type: str = "",
        aggregation_num: int = 50,
        print_workers: List[int] = [0],
        actor_id: int = 0,  # TODO
    ):
        if not self._is_load:
            return
        df = self.get_df()

        if plot_type == "":
            if self.is_mp:
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

            plt.plot(x, df["valid_reward"].rolling(rolling_n).mean(), "C0", label="valid_reward")
            for i in print_workers:
                plt.plot(x, df[f"episode_reward_{i}"].rolling(rolling_n).mean(), f"C{i+1}", label=f"reward {i}")

            alpha = 0.1
            plt.plot(x, df["valid_reward"], "C0", alpha=alpha)
            for i in print_workers:
                plt.plot(x, df[f"episode_reward_{i}"], f"C{i+1}", alpha=alpha)

            plt.xlabel(f"{xlabel}(mean {rolling_n})")
        else:
            plt.plot(x, df["valid_reward"], label="valid_reward")
            for i in print_workers:
                plt.plot(x, df[f"episode_reward_{i}"], label="reward")
            plt.xlabel(xlabel)

        plt.ylabel("reward")

        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_info(
        self,
        info_name,
        key,
        plot_type: str = "",
        aggregation_num: int = 50,
        actor_id: int = 0,  # TODO
    ) -> None:
        if not self._is_load:
            return
        df = self.get_df()

        if plot_type == "":
            if self.is_mp:
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

        key = f"{info_name}_{key}"
        if key not in df:
            return

        if len(df) > aggregation_num * 1.5:
            rolling_n = int(len(df) / aggregation_num)

            plt.plot(x, df[key].rolling(rolling_n).mean(), "C0")
            plt.plot(x, df[key], "C0", alpha=0.1)

            plt.xlabel(f"{xlabel}(mean {rolling_n})")
        else:
            plt.plot(x, df[key])
            plt.xlabel(f"{xlabel}")

        plt.ylabel(key)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def multi_plot(historys: List[Tuple[str, FileLogPlot]]):
    plt.xlabel("episode")
    plt.ylabel("valid reward")
    for name, h in historys:
        df = h.get_df()

        # timeを揃える
        # df = df.drop_duplicates(subset="time")
        # plt.plot(df["time"], df["valid_reward"], label=name)

        df = df.groupby("time").mean()
        plt.plot(df.index, df["valid_reward"], label=name)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
