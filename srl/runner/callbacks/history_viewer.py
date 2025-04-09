import json
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from srl.utils.common import is_package_installed, is_packages_installed

if TYPE_CHECKING:
    from srl.runner.callbacks.history_on_memory import HistoryOnMemory
    from srl.runner.runner_base import RunnerBase

logger = logging.getLogger(__name__)

"""
logs = [
    {
        "name" : trainer, actor0, actor1, ...
        "time" : 学習実行時からの経過時間

        # --- episode関係
        "step"         : 総ステップ数
        "episode"      : 総エピソード数
        "episode_step" : 1エピソードの総step
        "episode_time" : 1エピソードのtime
        "rewardX"      : 1エピソードの player の総報酬
        "eval_rewardX" : 評価してる場合はその報酬
        "sync"         : mpの時の同期回数
        "workerX_YYY"  : 学習情報

        # --- memory
        "memory" : memory に入っているbatch数

        # --- train関係
        "train"       : 学習回数
        "train_time"  : 区間内での学習時間の平均値
        "sync"        : mpの時の同期回数
        "trainer_YYY" : 学習情報

        # --- system関係
        "system_memory" : メモリ使用率
        "cpu"        : CPU使用率
        "gpuX"       : GPU使用率
        "gpuX_memory": GPUメモリの使用率
    },
    ...
]
"""


def _load_log_file(path: str) -> List[dict]:
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


class HistoryViewer:
    def __init__(self, save_dir: str = "") -> None:
        self.df = None

        if save_dir != "":
            self.load(save_dir)

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
            lines = _load_log_file(os.path.join(save_dir, f"actor{i}.txt"))
            self.logs.extend(lines)

        lines = _load_log_file(os.path.join(save_dir, "trainer.txt"))
        self.logs.extend(lines)

        lines = _load_log_file(os.path.join(save_dir, "system.txt"))
        self.logs.extend(lines)

        lines = _load_log_file(os.path.join(save_dir, "client.txt"))
        self.logs.extend(lines)

        # sort
        self.logs.sort(key=lambda x: x["time"])

    # ------------------------------------
    # memory
    # ------------------------------------
    def set_history_on_memory(self, history: "HistoryOnMemory", runner: "RunnerBase"):
        self.env_config: dict = runner.env_config.to_dict()
        self.rl_config: dict = runner.rl_config.to_dict()
        self.context: dict = runner.context.to_dict()
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
        _for_test: bool = False,
    ):
        ylabel_left = ylabel_left[:]
        ylabel_right = ylabel_right[:]

        assert is_packages_installed(["matplotlib", "pandas"]), "To use plot you need to install the 'matplotlib', 'pandas'. (pip install matplotlib pandas)"
        assert len(ylabel_left) > 0

        df = self.get_df()
        if len(df) == 0:
            s = "DataFrame length is 0."
            print(s)
            logger.info(s)
            return

        if xlabel not in df:
            s = f"'{xlabel}' is not found."
            print(s)
            logger.info(s)
            return

        n = 0
        for column in ylabel_left:
            if column in df:
                n += 1
        for column in ylabel_right:
            if column in df:
                n += 1
        if n == 0:
            s = f"'{ylabel_left}' '{ylabel_right}' is not found."
            print(s)
            logger.info(s)
            return

        # --- あるcolumnのみ
        ylabel_left = [t for t in ylabel_left if t in df.columns]
        ylabel_right = [t for t in ylabel_right if t in df.columns]

        _df = df[[xlabel] + ylabel_left + ylabel_right]
        _df = _df.dropna()
        if len(_df) == 0:
            s = "DataFrame length is 0."
            print(s)
            logger.info(s)
            return

        if len(_df) > aggregation_num * 2:
            rolling_n = int(len(_df) / aggregation_num)
            xlabel_plot = f"{xlabel} ({rolling_n}mean)"
        else:
            rolling_n = 0
            xlabel_plot = xlabel

        import matplotlib.pyplot as plt

        if _for_test:
            import matplotlib

            matplotlib.use("Agg")

        x = _df[xlabel]
        fig, ax1 = plt.subplots()
        color_idx = 0
        for column in ylabel_left:
            if column not in _df:
                continue
            if rolling_n > 0:
                ax1.plot(x, _df[column].rolling(rolling_n).mean(), f"C{color_idx}", label=column)
                ax1.plot(x, _df[column], f"C{color_idx}", alpha=0.2)
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

        if _for_test:
            plt.clf()
            plt.close()
            return

        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()


class HistoryViewers:
    def __init__(self, history_dirs: List[str]) -> None:
        self.histories: List[Tuple[str, HistoryViewer]] = []
        for d in history_dirs:
            if os.path.isdir(d):
                self.histories.append(
                    (
                        os.path.basename(d),
                        HistoryViewer(d),
                    )
                )

    def plot(
        self,
        xlabel: str = "time",
        ylabel: str = "reward0",
        aggregation_num: int = 50,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        title: str = "",
        _no_plot: bool = False,  # for test
    ):
        assert is_packages_installed(["matplotlib", "pandas"]), "To use plot you need to install the 'matplotlib', 'pandas'. (pip install matplotlib pandas)"

        import matplotlib.pyplot as plt

        plt.figure()
        color_idx = 0
        for name, h in self.histories:
            df = h.get_df()
            if len(df) == 0:
                s = f"[{name}] DataFrame length is 0."
                print(s)
                logger.info(s)
                continue

            if xlabel not in df:
                s = f"[{name}] '{xlabel}' is not found. {df.columns}"
                print(s)
                logger.info(s)
                continue

            if ylabel not in df:
                s = f"[{name}] '{ylabel}' is not found. {df.columns}"
                print(s)
                logger.info(s)
                continue

            _df = df[[xlabel, ylabel]]
            _df = _df.dropna()
            if len(_df) == 0:
                s = f"[{name}] DataFrame length is 0."
                print(s)
                logger.info(s)
                return

            if len(_df) > aggregation_num * 2:
                rolling_n = int(len(_df) / aggregation_num)
            else:
                rolling_n = 0

            _df.sort_values(xlabel, inplace=True)
            x = _df[xlabel]
            y = _df[ylabel]
            label = name.replace("_", " ").strip()
            if rolling_n > 0:
                plt.plot(x, y.rolling(rolling_n).mean(), f"C{color_idx}", label=label)
                plt.plot(x, y, f"C{color_idx}", alpha=0.2)
            else:
                plt.plot(x, y, f"C{color_idx}", label=label)
            color_idx += 1

            if ymin is not None:
                plt.ylim(bottom=ymin)
            if ymax is not None:
                plt.ylim(top=ymax)

        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        if title != "":
            plt.title(title)
        plt.tight_layout()
        if not _no_plot:
            plt.show()
        plt.clf()
        plt.close()
