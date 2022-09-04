import datetime as dt
import logging
import time
from dataclasses import dataclass

import numpy as np
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.utils.common import is_package_installed, listdictdict_to_dictlist, summarize_info_from_dictlist, to_str_time

logger = logging.getLogger(__name__)


# 進捗に対して表示、少しずつ間隔を長くする(上限あり)
@dataclass
class PrintProgress(Callback):

    max_time: int = 60 * 10  # s
    start_time: int = 5  # s

    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0

    max_actor: int = 5

    def __post_init__(self):
        assert self.start_time > 0
        assert self.start_time < self.max_time
        self.progress_timeout = self.start_time
        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

        self.last_episode_time = np.inf

    def on_episodes_begin(self, info):
        self.config = info["config"]
        self.actor_id = info["actor_id"]

        if self.actor_id >= self.max_actor:
            return

        if not self.config.distributed:
            print(
                "### env: {}, rl: {}, max episodes: {}, timeout: {}, max steps: {}, max train: {}".format(
                    self.config.env_config.name,
                    self.config.rl_config.getName(),
                    self.config.max_episodes,
                    to_str_time(self.config.timeout),
                    self.config.max_steps,
                    self.config.max_train_count,
                )
            )

        self.progress_t0 = self.t0 = time.time()
        self.progress_history = []

    def on_episodes_end(self, info):
        trainer = info["trainer"]

        if self.actor_id >= self.max_actor:
            return
        if trainer is None:
            train_count = 0
        else:
            train_count = trainer.get_train_count()
        self._print(info["episode_count"], train_count)

    def on_episode_begin(self, info):
        if self.actor_id >= self.max_actor:
            return
        self.history_step = []

    def on_step_end(self, info):
        if self.actor_id >= self.max_actor:
            return

        self.step_count += 1
        d = {
            "env_info": info["env"].info,
            "work_info": info["workers"][self.print_worker].info,
            "step_time": info["step_time"],
            "train_info": info["train_info"],
            "train_time": info["train_time"],
        }
        if self.actor_id == 0:
            remote_memory = info["remote_memory"]
            d["remote_memory"] = remote_memory.length() if remote_memory is not None else 0
        else:
            d["remote_memory"] = 0
        self.history_step.append(d)

        if self._check_print_progress():
            trainer = info["trainer"]
            if trainer is None:
                train_count = 0
            else:
                train_count = trainer.get_train_count()
            self._print(info["episode_count"], train_count)

    def on_episode_end(self, info):
        episode_time = info["episode_time"]
        if self.actor_id >= self.max_actor:
            return
        self.last_episode_time = episode_time
        if len(self.history_step) == 0:
            return
        player_idx = info["worker_indices"][self.print_worker]

        # 1エピソードの結果を平均でまとめる
        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        env_info = summarize_info_from_dictlist(env_info)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        work_info = summarize_info_from_dictlist(work_info)

        d = {
            "episode_step": info["episode_step"],
            "episode_reward": info["episode_rewards"][player_idx],
            "episode_time": episode_time,
            "eval_reward": info["eval_reward"],
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "remote_memory": self.history_step[-1]["remote_memory"],
            "env_info": env_info,
            "work_info": work_info,
            "train_time": np.mean([h["train_time"] for h in self.history_step]),
        }
        if "sync" in info:
            d["sync"] = info["sync"]

        # train info
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            d["train_info"] = summarize_info_from_dictlist(train_info)

        self.progress_history.append(d)

    # -----------------------------------------

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = _time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_time:
            self.progress_timeout = self.max_time

        return True

    def _print(self, episode_count, train_count):
        elapsed_time = time.time() - self.t0

        # --- head
        # [TIME] [actor] [elapsed time] [all step] [all episode] [train]
        s = dt.datetime.now().strftime("%H:%M:%S")
        if self.config.distributed:
            s += f" actor{self.actor_id:2d}:"
        s += f" {to_str_time(elapsed_time)}"
        s += f" {self.step_count:6d}st({episode_count:4d}ep)"
        if self.config.training and not self.config.distributed:
            s += " {:5d}tr".format(train_count)

        if len(self.progress_history) == 0:
            if len(self.history_step) == 0:
                # --- no info
                s += "1 step is not over."
            else:
                # --- steps info
                # [remain] [episode step] [step time] [train time] [memory]
                step_num = len(self.history_step)
                step_time = np.mean([h["step_time"] for h in self.history_step])
                if self.config.training and not self.config.distributed:
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
                # TODO max train count
                remain = min(min(remain_step, remain_episode), remain_time)
                s += f" {to_str_time(remain)}(remain)"

                # episode step, step time
                s += f", {step_num:5d} step"
                s += f", {step_time:.5f}s/step"

                # train time, memory
                if self.config.training and not self.config.distributed:
                    s += f", {train_time:.5f}s/tr"

                    memory_len = max([h["remote_memory"] for h in self.history_step])
                    s += self._memory_str(memory_len)

                # info
                if self.print_env_info:
                    s += self._info_str(self.history_step, "env_info")
                if self.print_worker_info:
                    s += self._info_str(self.history_step, "work_info")
                if self.print_train_info:
                    s += self._info_str(self.history_step, "train_info")

        else:
            # --- episode info
            # [remain] [reward] [eval reward] [episode step] [episode time] [train time] [sync] [memory] [info]
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
            # TODO max train count
            remain = min(min(remain_step, remain_episode), remain_time)
            s += f" {to_str_time(remain)}(remain)"

            # reward
            _r = [h["episode_reward"] for h in self.progress_history]
            s += f", {min(_r):.1f} {np.mean(_r):.3f} {max(_r):.1f} re"

            # eval reward
            if self.config.distributed:
                if self.config.enable_evaluation:
                    s += self._eval_reward_str(self.progress_history)
                else:
                    s += self._eval_reward_str(None)
            else:
                if self.config.enable_evaluation:
                    s += self._eval_reward_str(self.progress_history)

            # episode step, episode time
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {np.mean(_s):.1f} step"
            s += f", {episode_time:.2f}s/ep"

            # train time
            if self.config.training and not self.config.distributed:
                train_time = np.mean([h["train_time"] for h in self.progress_history])
                s += f", {train_time:.3f}s/tr"

            # sync
            if "sync" in self.progress_history[0]:
                sync = max([h["sync"] for h in self.progress_history])
                s += f", {sync:3d} recv"

            # memory
            if self.actor_id == 0:
                memory_len = max([h["remote_memory"] for h in self.progress_history])
                s += self._memory_str(memory_len)

            # info
            if self.print_env_info:
                s += self._info_str(self.progress_history, "env_info")
            if self.print_worker_info:
                s += self._info_str(self.progress_history, "work_info")
            if self.print_train_info:
                s += self._info_str(self.progress_history, "train_info")

        print(s)
        self.progress_history = []

    def _eval_reward_str(self, arr):
        if arr is None:
            return " " * 12
        _rewards = [h["eval_reward"] for h in arr if h["eval_reward"] is not None]
        if len(_rewards) > 0:
            s = f"({np.mean(_rewards):.3f} eval)"
        else:
            s = " " * 12
        return s

    def _memory_str(self, memory_len):
        # , 1234567 mem(100% PC)
        s = f", {memory_len:7d} mem"

        if is_package_installed("psutil"):
            import psutil

            s += f"({psutil.virtual_memory().percent:.1f}% PC)"

        return s

    def _info_str(self, arr_dict, key):
        s = ""
        d = listdictdict_to_dictlist(arr_dict, key)
        d = summarize_info_from_dictlist(d)
        for k, v in d.items():
            if v is None:
                s += f"|{k} None"
            elif isinstance(v, float):
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v}"
        return s


@dataclass
class TrainerPrintProgress(TrainerCallback):
    max_time: int = 60 * 10  # s
    start_time: int = 5  # s

    print_train_info: bool = True

    def __post_init__(self):
        assert self.start_time > 0
        assert self.start_time < self.max_time
        self.progress_timeout = self.start_time

    def on_trainer_start(self, info) -> None:
        self.config = info["config"]
        remote_memory = info["remote_memory"]

        if not self.config.distributed:
            print(
                "### max train: {}, timeout: {}, memory len: {}".format(
                    self.config.max_train_count,
                    to_str_time(self.config.timeout),
                    remote_memory.length(),
                )
            )

        self.progress_t0 = self.t0 = time.time()
        self.progress_history = []

        self.train_time = 0
        self.train_count = 0

    def on_trainer_end(self, info) -> None:
        self._print()

    def on_trainer_train(self, info) -> None:
        self.train_time = info["train_time"]
        self.train_count = info["train_count"]

        d = {
            "train_count": self.train_count,
            "train_time": self.train_time,
            "train_info": info["train_info"],
            "eval_reward": info["eval_reward"],
        }
        if "sync" in info:
            d["sync"] = info["sync"]

        self.progress_history.append(d)
        if self._check_print_progress():
            self._print()

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = _time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_time:
            self.progress_timeout = self.max_time

        return True

    def _print(self):
        elapsed_time = time.time() - self.t0

        # --- head
        # [TIME] [trainer] [elapsed time] [train count]
        s = dt.datetime.now().strftime("%H:%M:%S")
        if self.config.distributed:
            s += " trainer:"
        s += f" {to_str_time(elapsed_time)}"
        s += " {:6d}tr".format(self.train_count)

        if len(self.progress_history) == 0:
            # --- no info
            s += "1 train is not over."
        else:
            # --- train info
            # [remain] [train time] [eval] [sync] [info]
            train_time = np.mean([h["train_time"] for h in self.progress_history])

            if self.config.max_train_count > 0:
                remain_train = (self.config.max_train_count - self.train_count) * train_time
            else:
                remain_train = np.inf
            if self.config.timeout > 0:
                remain_time = self.config.timeout - elapsed_time
            else:
                remain_time = np.inf
            remain = min(remain_train, remain_time)
            s += f" {to_str_time(remain)}(remain)"

            # train time
            s += f", {train_time:.4f}s/tr"

            # eval
            if self.config.enable_evaluation:
                eval_rewards = [h["eval_reward"] for h in self.progress_history if h["eval_reward"] is not None]
                if len(eval_rewards) > 0:
                    s += f", {np.mean(eval_rewards):.3f} eval reward"
            # sync
            if "sync" in self.progress_history[0]:
                sync = max([h["sync"] for h in self.progress_history])
                s += f", {sync:3d} send"

            # train info
            if self.print_train_info:
                d = listdictdict_to_dictlist(self.progress_history, "train_info")
                d = summarize_info_from_dictlist(d)
                for k, v in d.items():
                    if v is None:
                        s += f"|{k} None"
                    elif isinstance(v, float):
                        s += f"|{k} {v:.4f}"
                    else:
                        s += f"|{k} {v}"

        print(s)
        self.progress_history = []


@dataclass
class MPPrintProgress(MPCallback):
    def on_start(self, info):
        config = info["config"]

        print(
            "### env: {}, rl: {}, max train: {}, timeout: {}".format(
                config.env_config.name,
                config.rl_config.getName(),
                config.max_train_count,
                to_str_time(config.timeout),
            )
        )
