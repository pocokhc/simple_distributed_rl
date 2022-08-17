import datetime as dt
import logging
import time
from dataclasses import dataclass

import numpy as np
from srl.base.env.base import EnvRun
from srl.runner.callback import Callback
from srl.utils.common import listdictdict_to_dictlist, to_str_time

logger = logging.getLogger(__name__)


# 進捗初期化、進捗に対して表示、少しずつ間隔を長くする(上限あり)
@dataclass
class PrintProgress(Callback):

    max_progress_time: int = 60 * 10  # s
    start_progress_timeout: int = 5  # s
    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0

    def __post_init__(self):
        assert self.start_progress_timeout > 0
        assert self.start_progress_timeout < self.max_progress_time
        self.progress_timeout = self.start_progress_timeout
        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

        self.last_episode_time = np.inf

    def on_episodes_begin(self, config, **kwargs):
        self.config = config
        print(
            "### env: {}, rl: {}, max episodes: {}, max steps: {}, timeout: {}".format(
                self.config.env_config.name,
                self.config.rl_config.getName(),
                self.config.max_episodes,
                self.config.max_steps,
                to_str_time(self.config.timeout),
            )
        )
        self.progress_t0 = self.t0 = time.time()
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
        worker_indices,
        **kwargs,
    ):
        self.last_episode_time = episode_time
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
            "remote_memory": self.history_step[-1]["remote_memory"],
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
        remote_memory,
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
            "remote_memory": remote_memory.length() if remote_memory is not None else 0,
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
        s += f" {self.step_count:6d}st({episode_count:4d}ep)"
        if self.config.training:
            s += " {:5d}tr".format(train_count)

        if len(self.progress_history) == 0:
            if len(self.history_step) > 0:
                step_num = len(self.history_step)
                step_time = np.mean([h["step_time"] for h in self.history_step])
                if self.config.training:
                    train_time = np.mean([h["train_time"] for h in self.history_step])
                else:
                    train_time = 0

                # remain
                if self.config.max_steps > 0:
                    remain_step = (self.config.max_steps - self.step_count) * (step_time + train_time)
                else:
                    remain_step = np.inf
                if self.config.max_episodes > 0:
                    remain_episode = (self.config.max_episodes - episode_count) * self.last_episode_time
                else:
                    remain_episode = np.inf
                if self.config.timeout > 0:
                    remain_time = self.config.timeout - elapsed_time
                else:
                    remain_time = np.inf
                remain = min(min(remain_step, remain_episode), remain_time)
                s += f" {to_str_time(remain)}(remain)"

                # steps info
                s += f", {step_num:5d} step"
                s += f", {step_time:.5f}s/step"
                if train_time > 0:
                    s += f", {train_time:.5f}s/tr"
                memory_len = max([h["remote_memory"] for h in self.history_step])
                s += f", {memory_len:7d} mem"
            else:
                s += "1 step is not over."

        else:
            episode_time = np.mean([h["episode_time"] for h in self.progress_history])

            # remain
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

            # episodes info
            _r = [h["episode_reward"] for h in self.progress_history]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {min(_r):.1f} {np.mean(_r):.3f} {max(_r):.1f} rew"
            s += f", {np.mean(_s):.1f} step"
            s += f", {episode_time:.2f}s/ep"

            if self.config.enable_validation:
                valid_rewards = [h["valid_reward"] for h in self.progress_history if h["valid_reward"] is not None]
                if len(valid_rewards) > 0:
                    s += f", {np.mean(valid_rewards):.3f} val_rew"

            if self.config.training:
                train_time = np.mean([h["train_time"] for h in self.progress_history])
                s += f", {train_time:.3f}s/tr"

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
