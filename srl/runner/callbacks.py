import datetime as dt
import io
import logging
import sys
import time
from abc import ABC
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from srl.base.define import RenderType
from srl.base.env.base import EnvRun

try:
    import matplotlib.pyplot as plt
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
    from matplotlib.animation import ArtistAnimation
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

try:
    from IPython import display
except ImportError:
    pass


from srl.utils.common import listdictdict_to_dictlist, to_str_time

logger = logging.getLogger(__name__)


class Callback(ABC):
    def on_episodes_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_episodes_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_episode_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_episode_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_step_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_step_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_skip_step(self, **kwargs) -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, **kwargs) -> bool:
        return False


@dataclass
class Rendering(Callback):

    mode: Union[str, RenderType] = RenderType.Terminal
    step_stop: bool = False
    enable_animation: bool = False

    def __post_init__(self):
        self.frames = []

        if isinstance(self.mode, str):
            for t in RenderType:
                if t.value == self.mode:
                    self.mode = t
                    break
            else:
                self.mode = RenderType.NONE

    def on_episode_begin(self, env: EnvRun, **kwargs):
        if self.mode != RenderType.NONE:
            print("### 0")
            env.render(self.mode)

        if self.enable_animation:
            self._add_image(env)

    def on_step_begin(
        self,
        env: EnvRun,
        workers,
        worker_indices,
        **kwargs,
    ) -> None:
        if self.mode != RenderType.NONE:
            for i in env.next_player_indices:
                worker_idx = worker_indices[i]
                workers[worker_idx].render(env)

        if self.step_stop:
            input("Enter to continue:")

    def on_step_end(
        self,
        env: EnvRun,
        actions,
        worker_indices,
        workers,
        train_info,
        **kwargs,
    ):

        if self.mode != RenderType.NONE:
            print(
                "### {}, actions {}, rewards {}, done {}({}), next {}".format(
                    env.step_num,
                    actions,
                    env.step_rewards,
                    env.done,
                    env.done_reason,
                    env.next_player_indices,
                )
            )
            env.render(self.mode)
            print(f"env_info  : {env.info}")
            for i in env.next_player_indices:
                worker_idx = worker_indices[i]
                print(f"work_info {worker_idx}: {workers[worker_idx].info}")
            print(f"train_info: {train_info}")

        if self.enable_animation:
            self._add_image(env)

    def on_skip_step(self, env: EnvRun, **kwargs):
        if self.mode != RenderType.NONE:
            env.render(self.mode)

        if self.enable_animation:
            self._add_image(env)

    # -----------------------------
    def _add_image(self, env: EnvRun):
        try:
            self.frames.append(env.render(RenderType.RGB_Array, is_except=True))
        except NotImplementedError:
            # --- printを画像に
            text = ""
            _stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                env.render(RenderType.Terminal)
                text = sys.stdout.getvalue()
            finally:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                sys.stdout = _stdout

            canvas_size = (300, 300)
            img = PIL.Image.new("RGB", canvas_size)
            draw = PIL.ImageDraw.Draw(img)
            text_width, text_height = draw.multiline_textsize(text)

            canvas_size = (text_width, text_height)
            background_rgb = (255, 255, 255)
            text_rgb = (0, 0, 0)
            img = PIL.Image.new("RGB", canvas_size, background_rgb)
            draw = PIL.ImageDraw.Draw(img)
            draw.text((0, 0), text, fill=text_rgb)
            self.frames.append(np.asarray(img))

    def create_anime(self, scale: float = 1.0, fps: float = 60):
        if len(self.frames) == 0:
            return None
        t0 = time.time()
        interval = 1000 / fps
        fig = plt.figure(figsize=(6.4 * scale, 4.8 * scale))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = []
        for f in self.frames:
            images.append([ax.imshow(f, animated=True)])
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook で画像が残るので出来ればcloseしたいけど、closeするとgym側でバグる
        logger.info("create animation({:.1f}s)".format(time.time() - t0))
        return anime

    def display(self, scale: float = 1.0, fps: float = 60) -> None:
        if len(self.frames) == 0:
            return
        t0 = time.time()
        anime = self.create_anime(scale, fps)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.info("create display({:.1f}s)".format(time.time() - t0))


# 進捗初期化、進捗に対して表示、少しずつ間隔を長くする(上限あり)
@dataclass
class PrintProgress(Callback):

    max_progress_time: int = 60 * 10  # s
    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0

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

    def on_episodes_begin(self, config, **kwargs):
        self.config = config
        print(
            "### env: {}, max episodes: {}, max steps: {}, timeout: {}".format(
                self.config.env_config.name,
                self.config.max_episodes,
                self.config.max_steps,
                to_str_time(self.config.timeout),
            )
        )
        self.progress_history = []

    def on_episodes_end(self, episode_count, trainer, **kwargs):
        if trainer is None:
            train_count = 0
        else:
            train_count = trainer.get_train_count()
        self._print(episode_count, train_count)

    def on_episode_begin(self, **kwargs):
        self.history_step = []

    def on_episode_end(
        self,
        episode_step,
        episode_rewards,
        episode_time,
        valid_reward,
        remote_memory,
        worker_indices,
        **kwargs,
    ):
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

        worker_idx = worker_indices[self.print_worker]
        d = {
            "episode_step": episode_step,
            "episode_reward": episode_rewards[worker_idx],
            "episode_time": episode_time,
            "valid_reward": valid_reward,
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "remote_memory": remote_memory.length() if remote_memory is not None else 0,
            "env_info": env_info,
            "work_info": work_info,
            "train_time": np.mean([h["train_time"] for h in self.history_step]),
        }

        # train info
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
            d["train_info"] = train_info

        self.progress_history.append(d)

    def on_step_end(
        self,
        env: EnvRun,
        episode_count,
        trainer,
        workers,
        train_info,
        step_time,
        train_time,
        **kwargs,
    ):
        self.step_count += 1
        d = {
            "env_info": env.info,
            "work_info": workers[self.print_worker].info,
            "train_info": train_info,
            "step_time": step_time,
            "train_time": train_time,
        }
        self.history_step.append(d)

        if self._check():
            if trainer is None:
                train_count = 0
            else:
                train_count = trainer.get_train_count()
            self._print(episode_count, train_count)

    def _check(self):

        # --- 時間経過したか
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

    def _print(self, episode_count, train_count):
        elapsed_time = time.time() - self.t0

        # --- print
        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f" {to_str_time(elapsed_time)}"
        s += " {:6d}ep".format(episode_count)
        if self.config.training:
            s += " {:6d}tr".format(train_count)

        if len(self.progress_history) == 0:
            if len(self.history_step) > 0:
                step_num = len(self.history_step)
                step_time = np.mean([h["step_time"] for h in self.history_step])
                s += f", {step_num:5d} step"
                s += f", {step_time:.5f}s/step"
                if self.config.training:
                    train_time = np.mean([h["train_time"] for h in self.history_step])
                    s += f", {train_time:.5f}s/tr"
        else:
            episode_time = np.mean([h["episode_time"] for h in self.progress_history])

            # 残り時間
            if self.config.max_steps > 0:
                step_time = np.mean([h["step_time"] for h in self.progress_history])
                train_time = np.mean([h["train_time"] for h in self.progress_history])
                remain_step = (self.config.max_steps - self.step_count) * (step_time + train_time)
            else:
                remain_step = np.inf
            if self.config.max_episodes > 0:
                remain_episode = (self.config.max_episodes - episode_count) * episode_time
            else:
                remain_episode = np.inf
            if self.config.timeout > 0:
                remain_time = self.config.timeout - elapsed_time
            else:
                remain_time = np.inf
            remain = min(min(remain_step, remain_episode), remain_time)
            s += f" {to_str_time(remain)}(remain)"

            # 表示
            _r = [h["episode_reward"] for h in self.progress_history]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {min(_r):.1f} {np.mean(_r):.3f} {max(_r):.1f} rew"
            s += f", {np.mean(_s):.1f} step"
            s += f", {episode_time:.3f}s/ep"

            if self.config.enable_validation:
                valid_rewards = [h["valid_reward"] for h in self.progress_history if h["valid_reward"] is not None]
                if len(valid_rewards) > 0:
                    s += f", {np.mean(valid_rewards):.3f} val_rew"

            if self.config.training:
                train_time = np.mean([h["train_time"] for h in self.progress_history])
                s += f", {train_time:.4f}s/tr"

                memory_len = max([h["remote_memory"] for h in self.progress_history])
                s += f", {memory_len:7d} mem"

            if self.print_env_info:
                d = listdictdict_to_dictlist(self.progress_history, "env_info")
                for k, arr in d.items():
                    s += f"|{k} {np.mean(arr):.3f}"
            if self.print_worker_info:
                d = listdictdict_to_dictlist(self.progress_history, "work_info")
                for k, arr in d.items():
                    s += f"|{k} {np.mean(arr):.3f}"
            if self.print_train_info:
                d = listdictdict_to_dictlist(self.progress_history, "train_info")
                for k, arr in d.items():
                    s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []


@dataclass
class History(Callback):

    target_worker: int = 0
    first_time_clip_num: int = 1

    def on_episodes_begin(self, config, **kwargs):
        self.history = []

    def on_episode_begin(self, **kwargs):
        self.history_step = []

    def on_step_end(self, env, workers, train_info, **kwargs):
        self.history_step.append(
            {
                "env_info": env.info,
                "work_info": workers[self.target_worker].info,
                "train_info": train_info,
            }
        )

    def on_episode_end(
        self,
        episode_count,
        episode_rewards,
        worker_indices,
        valid_reward,
        **kwargs,
    ):
        if len(self.history_step) == 0:
            return

        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        for k, v in env_info.items():
            env_info[k] = np.mean(v)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        for k, v in work_info.items():
            work_info[k] = np.mean(v)
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
        else:
            train_info = {}

        worker_idx = worker_indices[self.target_worker]
        self.history.append(
            {
                "episode": episode_count,
                "reward": episode_rewards[worker_idx],
                "valid_reward": valid_reward,
                "env_info": env_info,
                "work_info": work_info,
                "train_info": train_info,
            }
        )

    # ----------------
    def get_rewards(self) -> List[float]:
        return [h["reward"] for h in self.history]

    def get_valid_rewards(self) -> List[float]:
        return [h["valid_reward"] for h in self.history]

    def plot(self) -> None:
        rewards = [h["reward"] for h in self.history]
        valid_rewards = [h["valid_reward"] for h in self.history]

        rolling_n = int(len(self.history) / 100)

        if len(self.history) > 100:
            alpha = 0.2
            plt.plot(pd.Series(rewards).rolling(rolling_n).mean(), "C0", label=f"reward(mean{rolling_n})")
            plt.plot(
                pd.Series(valid_rewards).rolling(rolling_n).mean(),
                "C1",
                marker=".",
                label=f"valid reward(mean{rolling_n})",
            )
        else:
            alpha = 1
        plt.plot(rewards, "C0", alpha=alpha, label="reward")
        plt.plot(valid_rewards, "C1", alpha=alpha, label="valid reward", marker=".")

        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_info(self, info_name, key) -> None:
        d = listdictdict_to_dictlist(self.history, info_name + "_info")
        rolling_n = int(len(self.history) / 100)

        for target_key, arr in d.items():
            if len(arr) > self.first_time_clip_num:
                arr = arr[self.first_time_clip_num :]
            if key != target_key:
                continue
            if len(self.history) > 100:
                alpha = 0.2
                plt.plot(pd.Series(arr).rolling(rolling_n).mean(), label=f"mean {rolling_n}")
            else:
                alpha = 1
            plt.plot(arr, alpha=alpha, label=key)
            plt.ylabel(key)
            plt.xlabel("episode")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
