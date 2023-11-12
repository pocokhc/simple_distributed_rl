import json
import logging
import os
from typing import TYPE_CHECKING, List, Optional

from srl.utils.common import is_package_installed, is_packages_installed

if TYPE_CHECKING:
    from srl.runner.callbacks.history_on_memory import HistoryOnMemory
    from srl.runner.runner import Runner

logger = logging.getLogger(__name__)

"""
logs = [
    {
        "name" : trainer, actor0, actor1, ...
        "time" : 学習実行時からの経過時間

        # --- episode関係
        "episode"         : 総エピソード数
        "episode_step"    : 1エピソードの総step
        "episode_time"    : 1エピソードのtime
        "rewardX" : 1エピソードの player の総報酬
        "eval_rewardX"    : 評価してる場合はその報酬
        "sync"    : mpの時の同期回数
        "workerX_YYY"     : 学習情報

        # --- memory
        "memory" : memory に入っているbatch数

        # --- train関係
        "train"         : 学習回数
        "train_time"    : 区間内での学習時間の平均値
        "sync"    : mpの時の同期回数
        "trainer_YYY"   : 学習情報

        # --- system関係
        "system_memory" : メモリ使用率
        "cpu"    : CPU使用率
        "gpuX"       : GPU使用率
        "gpuX_memory": GPUメモリの使用率
    },
    ...
]
"""


class HistoryViewer:
    def __init__(self) -> None:
        self.df = None

    # ------------------------------------
    # file
    # ------------------------------------
    def load(self, save_dir: str):
        if not os.path.isdir(save_dir):
            logger.info(f"History folder is not found.({save_dir})")
            return

        # --- version
        path = os.path.join(save_dir, "version.txt")
        if os.path.isfile(path):
            with open(path) as f:
                v = f.read()

            import srl
            from srl.utils.common import compare_equal_version

            if not compare_equal_version(v, srl.__version__):
                logger.warning(f"SRL version is different({v} != {srl.__version__})")

        # --- file
        path = os.path.join(save_dir, "env_config.json")
        if os.path.isfile(path):
            with open(path) as f:
                self.env_config: dict = json.load(f)

        path = os.path.join(save_dir, "rl_config.json")
        if os.path.isfile(path):
            with open(path) as f:
                self.rl_config: dict = json.load(f)

        path = os.path.join(save_dir, "context.json")
        if os.path.isfile(path):
            with open(path) as f:
                self.context: dict = json.load(f)

        # --- load file
        self.logs = []
        for i in range(self.context["actor_num"]):
            lines = self._load_log_file(os.path.join(save_dir, "history", f"actor{i}.txt"))
            self.logs.extend(lines)

        lines = self._load_log_file(os.path.join(save_dir, "history", "trainer.txt"))
        self.logs.extend(lines)

        # sort
        self.logs.sort(key=lambda x: x["time"])

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
    def set_history_on_memory(self, history: "HistoryOnMemory", runner: "Runner"):
        self.env_config: dict = runner.env_config.to_dict()
        self.rl_config: dict = runner.rl_config.to_dict()
        self.context: dict = runner.context_controller.to_dict()
        self.logs: list = history.logs

    # ----------------------------------------
    # train logs
    # ----------------------------------------
    def get_df(self, is_preprocess: bool = True):
        if self.df is not None:
            return self.df

        assert is_package_installed("pandas"), "This run requires installation of 'pandas'. (pip install pandas)"
        import pandas as pd

        self.df = pd.DataFrame(self.logs)

        if is_preprocess:
            # いくつかの値は間を埋める
            if "episode" in self.df:
                self.df["episode"] = self.df["episode"].interpolate(limit_direction="both")
                self.df["episode"] = self.df["episode"].astype(int)
            if "train" in self.df:
                self.df["train"] = self.df["train"].interpolate(limit_direction="both")
                self.df["train"] = self.df["train"].astype(int)
            if "memory" in self.df:
                self.df["memory"] = self.df["memory"].interpolate(limit_direction="both")
                self.df["memory"] = self.df["memory"].astype(int)

        return self.df

    def plot(
        self,
        xlabel: str = "time",
        ylabel_left: List[str] = ["reward0"],
        ylabel_right: List[str] = [],
        aggregation_num: int = 50,
        left_ymin: Optional[float] = None,
        left_ymax: Optional[float] = None,
        right_ymin: Optional[float] = None,
        right_ymax: Optional[float] = None,
        _no_plot: bool = False,  # for test
    ):
        ylabel_left = ylabel_left[:]
        ylabel_right = ylabel_right[:]

        assert is_packages_installed(
            ["matplotlib", "pandas"]
        ), "To use plot you need to install the 'matplotlib', 'pandas'. (pip install matplotlib pandas)"
        assert len(ylabel_left) > 0

        import matplotlib.pyplot as plt

        df = self.get_df()
        if len(df) == 0:
            logger.info("DataFrame length is 0.")
            return

        if xlabel not in df:
            logger.info(f"'{xlabel}' is not found.")
            return

        n = 0
        for column in ylabel_left:
            if column in df:
                n += 1
        for column in ylabel_right:
            if column in df:
                n += 1
        if n == 0:
            logger.info(f"'{ylabel_left}' '{ylabel_right}' is not found.")
            return

        _df = df[[xlabel] + ylabel_left + ylabel_right]
        _df = _df.dropna()
        if len(_df) == 0:
            logger.info("DataFrame length is 0.")
            return

        if len(_df) > aggregation_num * 2:
            rolling_n = int(len(_df) / aggregation_num)
            xlabel_plot = f"{xlabel} ({rolling_n}mean)"
        else:
            rolling_n = 0
            xlabel_plot = xlabel

        x = _df[xlabel]
        fig, ax1 = plt.subplots()
        color_idx = 0
        for column in ylabel_left:
            if column not in _df:
                continue
            if rolling_n > 0:
                ax1.plot(x, _df[column].rolling(rolling_n).mean(), f"C{color_idx}", label=column)
                ax1.plot(x, _df[column], f"C{color_idx}", alpha=0.1)
            else:
                ax1.plot(x, _df[column], f"C{color_idx}", label=column)
            color_idx += 1
        ax1.legend(loc="upper left")
        if left_ymin is not None:
            ax1.set_ylim(bottom=left_ymin)
        if left_ymax is not None:
            ax1.set_ylim(top=left_ymax)

        if len(ylabel_right) > 0:
            ax2 = ax1.twinx()
            for column in ylabel_right:
                if column not in _df:
                    continue
                if rolling_n > 0:
                    ax2.plot(x, _df[column].rolling(rolling_n).mean(), f"C{color_idx}", label=column)
                    ax2.plot(x, _df[column], f"C{color_idx}", alpha=0.1)
                else:
                    ax2.plot(x, _df[column], f"C{color_idx}", label=column)
                color_idx += 1
            ax2.legend(loc="upper right")
            if right_ymin is not None:
                ax1.set_ylim(bottom=right_ymin)
            if right_ymax is not None:
                ax1.set_ylim(top=right_ymax)

        ax1.set_xlabel(xlabel_plot)
        plt.grid()
        plt.tight_layout()
        if not _no_plot:
            plt.show()
