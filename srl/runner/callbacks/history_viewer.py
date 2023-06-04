import json
import logging
import os
from typing import List, Optional, cast

from srl.utils.common import compare_equal_version, is_package_installed, is_packages_installed

logger = logging.getLogger(__name__)

"""
# sequence
logs = [
    {
        "index"    : index
        "time" : 学習実行時からの経過時間
        "train"         : 学習回数
        "trainer_YYY"   : 学習情報
        "remote_memory" : remote_memory に入っているbatch数
        "actor0_episode"         : 総エピソード数
        "actor0_episode_step"    : 1エピソードの総step
        "actor0_episode_time"    : 1エピソードのtime
        "actor0_episode_rewardX" : 1エピソードの player の総報酬
        "actor0_eval_rewardX"    : 評価してる場合はその報酬
        "actor0_workerX_YYY"     : 学習情報

        # --- system関係
        "memory" : メモリ使用率
        "cpu"    : CPU使用率
        "cpu_X"  : CPU使用率の詳細
        "gpu"       : GPU使用率
        "gpu_memory": GPUメモリの使用率
    },
    ...

# distribute
logs = [
    {
        "index"    : 各ファイルを紐づけるindex
        "time_trainer" : 学習実行時からの経過時間
        "time_actorX"  : 学習実行時からの経過時間
        "time_system"  : 学習実行時からの経過時間

        # --- train関係
        "train"         : 学習回数
        "train_time"    : 区間内での学習時間の平均値
        "train_sync"    : mpの時の同期回数
        "remote_memory" : remote_memory に入っているbatch数
        "trainer_YYY"   : 学習情報

        # --- actor関係
        "actorX_episode"         : 総エピソード数
        "actorX_episode_step"    : 1エピソードの総step
        "actorX_episode_time"    : 1エピソードのtime
        "actorX_episode_rewardX" : 1エピソードの player の総報酬
        "actorX_eval_rewardX"    : 評価してる場合はその報酬
        "actorX_workerX_YYY"     : 学習情報
        "actorX_sync"         : mpの時の同期回数

        # --- system関係
        "memory" : メモリ使用率
        "cpu"    : CPU使用率
        "cpu_X"  : CPU使用率の詳細
        "gpu"       : GPU使用率
        "gpu_memory": GPUメモリの使用率
    },
    ...
]
"""


class HistoryViewer:
    def __init__(self) -> None:
        self.df = None
        self.log_dir = ""
        self.train_log_dir = ""
        self.param_dir = ""
        self.history_on_memory = None

        # config
        self.player_num = 1
        self.actor_num = 1
        self.distributed = False

    # ------------------------------------
    # file
    # ------------------------------------
    def set_dir(self, log_dir: str):
        self.log_dir = log_dir

    def load(self, log_dir: str):
        import glob

        if not os.path.isdir(log_dir):
            logger.info(f"Log folder is not found.({log_dir})")
            return

        self.train_log_dir = os.path.join(log_dir, "train_log")
        self.param_dir = os.path.join(log_dir, "params")

        # --- version
        path = os.path.join(log_dir, "version.txt")
        if os.path.isfile(path):
            with open(path) as f:
                v = f.read()

            import srl

            if not compare_equal_version(v, srl.__version__):
                logger.warning(f"log version is different({v} != {srl.__version__})")

        # --- config
        path = os.path.join(log_dir, "config.json")
        if os.path.isfile(path):
            with open(path) as f:
                d = json.load(f)
            self.player_num = d["env_config"]["player_num"]
            self.distributed = d["_distributed"]
            self.actor_num = d.get("actor_num", 1)

        # --- episode
        self.episode_files = glob.glob(os.path.join(log_dir, "episode*.txt"))
        self.episode_cache = {}

        # --- load file df
        import pandas as pd

        if self.distributed:
            # trainer
            train_logs = self._load_log_file(os.path.join(self.train_log_dir, "trainer.txt"))
            if len(train_logs) == 0:
                return
            df = pd.DataFrame(train_logs)
            df.drop_duplicates(subset="index", keep="last", inplace=True)
            df.rename(columns={"time": "time_trainer"}, inplace=True)

            # actor
            for i in range(self.actor_num):
                try:
                    actor_logs = self._load_log_file(os.path.join(self.train_log_dir, f"actor{i}.txt"))
                    if len(actor_logs) == 0:
                        continue
                    df_actor = pd.DataFrame(actor_logs)
                    df_actor.drop_duplicates(subset="index", keep="last", inplace=True)
                    rename = {}
                    for k in df_actor.columns:
                        if k in ["index", "time"]:
                            continue
                        rename[k] = f"actor{i}_{k}"
                    df_actor.rename(columns=rename, inplace=True)
                    df_actor.rename(columns={"time": f"time_actor{i}"}, inplace=True)
                    if len(df) == 0:
                        df = df_actor
                    else:
                        df = pd.merge(df, df_actor, on="index", how="outer")
                except Exception:
                    import traceback

                    logger.info(traceback.format_exc())

        else:
            # 逐次
            actor_logs = self._load_log_file(os.path.join(self.train_log_dir, "actor0.txt"))
            df = pd.DataFrame(actor_logs)
            rename = {}
            for k in df.columns:
                if "trainer_" in k:
                    continue
                if k in [
                    "index",
                    "time",
                    "remote_memory",
                    "sync",
                    "train",
                ]:
                    continue
                rename[k] = f"actor0_{k}"
            df.rename(columns=rename, inplace=True)
            df.drop_duplicates(subset="index", keep="last", inplace=True)

        # system
        try:
            system_logs = self._load_log_file(os.path.join(self.train_log_dir, "system.txt"))
            if len(system_logs) > 0:
                df_system = pd.DataFrame(system_logs)
                df_system.rename(columns={"time": "time_system"}, inplace=True)
                df = pd.merge(df, df_system, on="index", how="outer")
        except Exception:
            import traceback

            logger.info(traceback.format_exc())

        self.df = df
        return df

    def _load_log_file(self, path: str) -> List[dict]:
        if not os.path.isfile(path):
            return []
        import json

        data = []
        with open(path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    data.append(d)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError {e.args[0]}, '{line.strip()}'")
        return data

    # ------------------------------------
    # memory
    # ------------------------------------
    def set_memory(self, config, history_on_memory):
        from srl.runner import Config
        from srl.runner.callbacks.history_on_memory import HistoryOnMemory

        config = cast(Config, config)
        self.player_num = config.env_config.player_num
        self.actor_num = config.actor_num
        self.distributed = config.distributed

        self.history_on_memory = cast(HistoryOnMemory, history_on_memory)

    def _load_memory_df(self):
        if self.history_on_memory is None:
            return None

        import pandas as pd

        df = pd.DataFrame(self.history_on_memory.logs)
        return df

    # ----------------------------------------
    # train logs
    # ----------------------------------------
    def get_df(self):
        if self.df is not None:
            return self.df

        assert is_package_installed("pandas"), "This run requires installation of 'pandas'. (pip install pandas)"
        import pandas as pd

        if self.history_on_memory is not None:
            self.df = self._load_memory_df()
        if self.df is None:
            self.df = self.load(self.log_dir)
        if self.df is None:
            return pd.DataFrame()
        else:
            return self.df

    def plot(
        self,
        plot_left: List[str] = ["actor0_episode_reward0", "actor0_eval_reward0"],
        plot_right: List[str] = [],
        plot_type: str = "",
        aggregation_num: int = 50,
        left_ymin: Optional[float] = None,
        left_ymax: Optional[float] = None,
        right_ymin: Optional[float] = None,
        right_ymax: Optional[float] = None,
        _no_plot: bool = False,  # for test
    ):
        plot_left = plot_left[:]
        plot_right = plot_right[:]

        assert is_packages_installed(
            ["matplotlib", "pandas"]
        ), "To use plot you need to install the 'matplotlib', 'pandas'. (pip install matplotlib pandas)"
        assert len(plot_left) > 0

        import matplotlib.pyplot as plt

        df = self.get_df()
        if len(df) == 0:
            logger.info("DataFrame length is 0.")
            return

        if plot_type == "":
            if self.distributed:
                plot_type = "timeline"
            else:
                plot_type = "episode"

        if plot_type == "timeline":
            x = df["time_trainer"]
            xlabel = "time"
        else:
            df = df.drop_duplicates(subset="actor0_episode")
            x = df["actor0_episode"]
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
        if not _no_plot:
            plt.show()
