import datetime as dt
import logging
import time
from abc import ABC
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib import animation
from srl.utils.common import listdictdict_to_dictlist, to_str_time

logger = logging.getLogger(__name__)


class Callback(ABC):
    def on_episodes_begin(self, info) -> None:
        pass

    def on_episodes_end(self, info) -> None:
        pass

    def on_episode_begin(self, info) -> None:
        pass

    def on_episode_end(self, info) -> None:
        pass

    def on_step_begin(self, info) -> None:
        pass

    def on_step_end(self, info) -> None:
        pass

    def on_skip_step(self, info) -> None:
        pass

    # 外部から途中停止用
    def intermediate_stop(self, info) -> bool:
        return False


@dataclass
class Rendering(Callback):

    step_stop: bool = False

    def on_episode_begin(self, info):
        env = info["env"]
        worker = info["worker"]
        state = info["state"]
        valid_actions = info["valid_actions"]

        print("### 0")
        env.render()
        worker.render(state, valid_actions, env.action_to_str)

        if self.step_stop:
            input("Enter to continue:")

    def on_step_end(self, info):
        env = info["env"]
        worker = info["worker"]
        step = info["step"]
        action = info["action"]
        reward = info["reward"]
        done = info["done"]
        state = info["state"]
        valid_actions = info["valid_actions"]
        env_info = info["env_info"]
        work_info = info["work_info"]
        train_info = info["train_info"]

        print("### {}, action {}, reward: {}, done: {}".format(step, action, reward, done))
        print(f"env_info  : {env_info}")
        print(f"work_info : {work_info}")
        print(f"train_info: {train_info}")
        env.render()
        worker.render(state, valid_actions, env.action_to_str)

        if self.step_stop:
            input("Enter to continue:")

    def on_skip_step(self, info):
        info["env"].render()


class RenderingAnimation(Callback):
    def __init__(self):
        self.frames = []

    def on_episode_begin(self, info):
        env = info["env"]
        self.frames.append(env.render("rgb_array"))

    def on_step_end(self, info):
        env = info["env"]
        self.frames.append(env.render("rgb_array"))

    def on_skip_step(self, info):
        env = info["env"]
        self.frames.append(env.render("rgb_array"))

    # -------------------------------
    def create_anime(self, scale: float = 1.0, fps: int = 60) -> animation.ArtistAnimation:
        if len(self.frames) == 0:
            return None
        interval = 1000 / fps
        fig = plt.figure(figsize=(6.4 * scale, 4.8 * scale))
        plt.axis("off")
        images = []
        for f in self.frames:
            images.append([plt.imshow(f, animated=True)])
        anime = animation.ArtistAnimation(fig, images, interval=interval, repeat=False)
        plt.close()
        return anime

    def display(self, scale: float = 1.0, fps: int = 60):
        if len(self.frames) == 0:
            return
        t0 = time.time()
        anime = self.create_anime(scale, fps)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.debug("create movie({:.1f}s)".format(time.time() - t0))


# 進捗初期化、進捗に対して表示、少しずつ間隔を長くする(上限あり)
@dataclass
class PrintProgress(Callback):

    max_progress_time: int = 60 * 10  # s

    def __post_init__(self):
        self.progress_timeout = 5

        self.progress_t0 = self.t0 = time.time()
        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

        self.max_episodes = -1
        self.max_steps = -1
        self.timeout = -1

    def on_episodes_begin(self, info):
        self.max_episodes = info["config"].max_episodes
        self.max_steps = info["config"].max_steps
        self.timeout = info["config"].timeout
        print(
            f"### max episodes: {self.max_episodes}, max steps: {self.max_steps}, timeout: {to_str_time(self.timeout)}"
        )

        self.progress_history = []

    def on_episodes_end(self, info):
        self._print(info, is_last=True)

        if False:
            d = self._listdict_to_dictlist(self.history_episode)
            rolling_n = int(len(d) / 100)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            if len(d) > 100:
                alpha = 0.3
                ax1.plot(pd.Series(d["reward"]).rolling(rolling_n).mean(), "C1", label=f"reward(mean{rolling_n})")
            else:
                alpha = 1
            ax1.plot(d["reward"], "C0", alpha=alpha, label="reward")

            ax2 = ax1.twinx()
            if len(d) > 100:
                alpha = 0.3
                ax2.plot(pd.Series(d["step"]).rolling(rolling_n).mean(), "C3", label=f"reward(mean{rolling_n})")
            else:
                alpha = 1
            ax2.plot(d["step"], "C2", alpha=alpha, label="step")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="lower right")

            ax1.set_ylabel("reward")
            ax1.grid(True)
            ax2.set_ylabel("step")

            plt.tight_layout()
            plt.show()

            for key, arr in d.items():
                if key in ["reward", "step"]:
                    continue
                if len(d) > 100:
                    plt.plot(pd.Series(arr).rolling(rolling_n).mean(), label=f"mean {rolling_n}")
                    alpha = 0.3
                else:
                    alpha = 1
                plt.plot(arr, alpha=alpha, label="raw")
                plt.title(key)
                plt.grid()
                plt.legend()
                plt.tight_layout()
                plt.show()

    def on_episode_begin(self, info):
        self.history_step = []

    def on_episode_end(self, info):
        if len(self.history_step) == 0:
            return

        # 1エピソードの結果を平均でまとめる
        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        if "TimeLimit.truncated" in env_info:
            del env_info["TimeLimit.truncated"]
        for k, v in env_info.items():
            env_info[k] = np.mean(v)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        for k, v in work_info.items():
            work_info[k] = np.mean(v)

        d = {
            "step": info["step"],
            "reward": info["reward"],
            "episode_time": info["episode_time"],
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "remote_memory": info["remote_memory"].length(),
            "env_info": env_info,
            "work_info": work_info,
            "train_time": np.mean([h["step_time"] for h in self.history_step]),
        }

        # train info
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
            d["train_info"] = train_info

        self.progress_history.append(d)

    def on_step_end(self, info):
        d = {
            "env_info": info["env_info"],
            "work_info": info["work_info"],
            "train_info": info["train_info"],
            "step_time": info["step_time"],
            "train_time": info["train_time"],
        }
        if info["train_info"] is not None:
            d["train_info"] = info["train_info"]
        self.history_step.append(d)
        self._print(info)

    def _print(self, info, is_last=False):

        # --- 時間経過したか
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout and not is_last:
            return
        self.progress_t0 = _time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_progress_time:
            self.progress_timeout = self.max_progress_time

        self.elapsed_time = _time - self.t0

        # --- print
        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f" {to_str_time(self.elapsed_time)}"
        s += " {:7d}ep".format(info["episode_count"])
        if info["trainer"] is not None:
            s += " {:7d}tr".format(info["trainer"].get_train_count())

        if len(self.progress_history) > 0:
            episode_time = np.mean([h["episode_time"] for h in self.progress_history])
            step_time = np.mean([h["step_time"] for h in self.progress_history])

            # 残り時間
            if self.max_steps > 0:
                remain_step = (self.max_steps - self.step_count) * step_time
            else:
                remain_step = np.inf
            if self.max_episodes > 0:
                remain_episode = (self.max_episodes - info["episode_count"]) * episode_time
            else:
                remain_episode = np.inf
            if self.timeout > 0:
                remain_time = self.timeout - self.elapsed_time
            else:
                remain_time = np.inf
            remain = min(min(remain_step, remain_episode), remain_time)
            s += f" {to_str_time(remain)}(remain)"

            # 表示
            _r = [h["reward"] for h in self.progress_history]
            _s = [h["step"] for h in self.progress_history]
            s += f", {min(_r):.3f} {np.mean(_r):.3f} {max(_r):.3f} reward"
            s += f", {np.mean(_s):.1f} step"
            s += f", {episode_time:.2f}s/ep"

            train_time = np.mean([h["train_time"] for h in self.progress_history])
            s += f", {train_time:.4f}s/tr"

            memory_len = max([h["remote_memory"] for h in self.progress_history])
            s += f", {memory_len:8d} mem"

            d = listdictdict_to_dictlist(self.progress_history, "env_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"
            d = listdictdict_to_dictlist(self.progress_history, "work_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"
            d = listdictdict_to_dictlist(self.progress_history, "train_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []


if __name__ == "__main__":
    pass
