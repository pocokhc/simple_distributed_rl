import datetime as dt
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
    def load(self, path: str):
        self.set_path(path)
        self._read_df()

    def set_path(self, path: str):
        assert os.path.isdir(path)
        self.base_dir = path
        self.param_dir = os.path.join(path, "params")
        self.log_dir = os.path.join(path, "logs")
        self.df = None

        self.is_mp = os.path.isfile(os.path.join(self.base_dir, "mp_config.json"))

        with open(os.path.join(self.base_dir, "version.txt")) as f:
            v = f.read()
            if v != srl.__version__:
                logger.warning(f"log version is different({v} != {srl.__version__})")

        with open(os.path.join(self.base_dir, "config.json")) as f:
            d = json.load(f)
            self.player_num = d["env_config"]["player_num"]

        if self.is_mp:
            with open(os.path.join(self.base_dir, "mp_config.json")) as f:
                d = json.load(f)
                self.actor_num = d["actor_num"]

    def _read_log(self, filename):
        data = []
        with open(os.path.join(self.log_dir, filename)) as f:
            for line in f:
                data.append(json.loads(line))
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
        self.df.sort_values("date", inplace=True)
        self.df["time"] = (self.df["date"] - self.df["date"][0]).dt.total_seconds()
        self.df.set_index("date", inplace=True)

    # ----------------------------------------
    def get_df(self) -> pd.DataFrame:
        self._read_df()
        assert self.df is not None
        return self.df

    def plot(
        self,
        plot_type: str = "",
        aggregation_num: int = 50,
        print_workers: List[int] = [0],
        actor_id: int = 0,
    ):

        if plot_type == "":
            if self.is_mp:
                plot_type = "timeline"
            else:
                plot_type = "episode"

        self._read_df(actor_id)
        assert self.df is not None

        # nanを補完
        df = self.df.interpolate("time")

        if plot_type == "timeline":
            df = df.drop_duplicates(subset="time")
            x = df["time"]
            xlabel = "time"
        else:
            x = df["episode_count"]
            xlabel = "episode"

        if len(df) > aggregation_num * 2:
            rolling_n = int(len(df) / aggregation_num)

            plt.plot(x, df["valid_reward"].rolling(rolling_n).mean(), "C0", label=f"valid_reward")
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
    ) -> None:
        self._read_df()
        assert self.df is not None
        print(self.df)
        print(self.df.info())

        # nanを補完
        df = self.df.interpolate("time")

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

        if len(df) > aggregation_num * 2:
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
        # nanを補完
        df = h.get_df().interpolate("time")

        # timeを揃える
        # df = df.drop_duplicates(subset="time")
        # plt.plot(df["time"], df["valid_reward"], label=name)

        df = df.groupby("time").mean()
        plt.plot(df.index, df["valid_reward"], label=name)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
