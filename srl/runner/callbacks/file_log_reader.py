import glob
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

import srl
from srl.utils.common import compare_equal_version, is_package_installed

logger = logging.getLogger(__name__)


class FileLogReader:
    def __init__(self) -> None:
        self.base_dir = ""

    def load(self, path: str):
        if not os.path.isdir(path):
            logger.info(f"Log folder is not found.({path})")
            return

        self.base_dir = path
        self.train_log_dir = os.path.join(path, "train_log")
        self.episode_log_dir = os.path.join(path, "episode_log")
        self.param_dir = os.path.join(path, "params")
        self.path_config = os.path.join(self.base_dir, "config.json")
        self.path_system = os.path.join(self.base_dir, "system.json")
        self.path_version = os.path.join(self.base_dir, "version.txt")

        # config
        self.player_num = 1
        self.actor_num = 1
        self.distributed = False

        # local vals
        self.df = None
        self.logs = None

        # --- read config
        if os.path.isfile(self.path_version):
            with open(self.path_version) as f:
                v = f.read()
            if not compare_equal_version(v, srl.__version__):
                logger.warning(f"log version is different({v} != {srl.__version__})")

        if os.path.isfile(self.path_config):
            with open(self.path_config) as f:
                d = json.load(f)
            self.player_num = d["env_config"]["player_num"]
            self.distributed = d["distributed"]
            self.actor_num = d.get("actor_num", 1)

        # --- episode
        self.episode_files = glob.glob(os.path.join(self.episode_log_dir, "episode*.txt"))
        self.episode_cache = {}

    def remove_dir(self):
        # --- logs
        for fn in glob.glob(os.path.join(self.train_log_dir, "*.txt")):
            if os.path.isfile(fn):
                logger.debug(f"remove file: {fn}")
                os.remove(fn)
        logger.debug(f"remove dir : {self.train_log_dir}")
        os.rmdir(self.train_log_dir)  # 空のみ対象

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
            self.path_system,
            self.path_version,
        ]:
            if os.path.isfile(fn):
                logger.debug(f"remove file: {fn}")
                os.remove(fn)

        # dir
        logger.debug(f"remove dir : {self.base_dir}")
        os.rmdir(self.base_dir)  # 空のみ対象

    def _read_log(self, path: str) -> List[dict]:
        if not os.path.isfile(path):
            return []
        data = []
        with open(path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    data.append(d)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError {e.args[0]}, '{line.strip()}'")
        return data

    # ----------------------------------------
    # train logs
    # ----------------------------------------
    def _read_train_logs(self):
        if self.logs is not None:
            return

        assert os.path.isdir(
            self.train_log_dir
        ), f"'{self.train_log_dir}' is not found. Setting `enable_file_logger=True` in `runner.train` may solve this problem."

        self.logs = []

        # trainer
        train_logs = self._read_log(os.path.join(self.train_log_dir, "trainer.txt"))
        for t in train_logs:
            t["_type"] = "trainer"
        self.logs.extend(train_logs)

        # actor
        for i in range(self.actor_num):
            actor_logs = self._read_log(os.path.join(self.train_log_dir, f"actor{i}.txt"))
            for t in actor_logs:
                t["_type"] = f"actor{i}"
            self.logs.extend(actor_logs)

        # system
        system_logs = self._read_log(os.path.join(self.train_log_dir, "system.txt"))
        for t in system_logs:
            t["_type"] = "system"
        self.logs.extend(system_logs)

    def get_logs(self) -> List[dict]:
        self._read_train_logs()

        if self.logs is None:
            return []
        return self.logs

    def get_df(self):
        assert is_package_installed("pandas"), "To use get_df you need to install the 'pandas'. (pip install pandas)"
        self._read_train_logs()

        if self.df is not None:
            return self.df

        import pandas as pd

        df = pd.DataFrame(self.logs)

        if len(df) == 0:
            return df

        df["date"] = pd.to_datetime(df["date"])
        if "episode_count" in df:
            df.sort_values(["date", "episode_count"], inplace=True)
        else:
            df.sort_values(["date"], inplace=True)
        df["time"] = (df["date"] - df["date"][0]).dt.total_seconds()
        df.set_index("date", inplace=True)

        # nanを補完
        df = df.interpolate(method="time")

        self.df = df
        return self.df

    def plot(
        self,
        plot_left: List[str] = ["episode_reward0", "eval_reward0"],
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
            logger.info("DataFrame length is 0.")
            return

        if "episode_count" not in df:
            plot_type = "timeline"
        elif plot_type == "":
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

    # ----------------------------------------
    # episode logs
    # ----------------------------------------
    @property
    def episode_num(self) -> int:
        return len(self.episode_files)

    def load_episode(self, episode: int) -> Tuple[dict, List[dict]]:
        if episode < 0:
            return {}, []
        if episode >= len(self.episode_files):
            return {}, []

        if episode in self.episode_cache:
            return self.episode_cache[episode]

        logger.debug("Start loading episode.")
        t0 = time.time()

        steps = self._read_log(self.episode_files[episode])
        total_reward = np.zeros(self.player_num)
        for step in steps:
            if "rewards" in step:
                step["rewards"] = np.array(step["rewards"])
                total_reward += step["rewards"]
            if "env_rgb_array" in step:
                step["env_rgb_array"] = np.array(step["env_rgb_array"])
            for i in range(self.player_num):
                if f"work{i}_rgb_array" in step:
                    step[f"work{i}_rgb_array"] = np.array(step[f"work{i}_rgb_array"])
        episode_info = {
            "total_rewards": total_reward,
        }
        self.episode_cache[episode] = (episode_info, steps)

        logger.info(f"Episode loaded.({time.time()-t0:.1f}s)")
        return episode_info, steps

    def replay(self):
        from srl.runner.game_window import EpisodeReplay

        EpisodeReplay(self).play()


def load_history(log_dir: str) -> FileLogReader:
    history = FileLogReader()
    history.load(log_dir)
    return history
