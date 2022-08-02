import datetime as dt
import logging
import time
from dataclasses import dataclass

import numpy as np
from srl.runner.callback_mp import MPCallback
from srl.utils.common import listdictdict_to_dictlist, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class MPPrintProgress(MPCallback):

    max_progress_time: int = 60 * 10  # s
    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0
    max_print_actor: int = 5

    def __post_init__(self):
        self.progress_timeout = 5

    def _check_print_progress(self):

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

    # ---------------------------
    # main
    # ---------------------------
    def on_start(self, config, mp_config, **kwargs):
        print(
            "### env: {}, max train: {}, timeout: {}".format(
                config.env_config.name,
                mp_config.max_train_count,
                to_str_time(mp_config.timeout),
            )
        )

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, config, mp_config, **kwargs):
        self.max_train_count = mp_config.max_train_count
        self.timeout = mp_config.timeout

        self.progress_t0 = self.t0 = time.time()
        self.progress_history = []

        self.train_time = 0
        self.train_count = 0

    def on_trainer_end(self, **kwargs):
        self._trainer_print_progress()

    def on_trainer_train(
        self,
        remote_memory,
        train_count,
        train_time,
        train_info,
        sync_count,
        **kwargs,
    ):
        self.train_time = train_time
        self.train_count = train_count
        self.progress_history.append(
            {
                "train_count": train_count,
                "train_time": train_time,
                "train_info": train_info,
                "sync_count": sync_count,
                "remote_memory": remote_memory.length(),
            }
        )
        if self._check_print_progress():
            self._trainer_print_progress()

    def _trainer_print_progress(self):

        # --- 残り時間
        s = dt.datetime.now().strftime("%H:%M:%S")
        elapsed_time = time.time() - self.t0
        s += f" --- {to_str_time(elapsed_time)}"

        if self.max_train_count > 0:
            remain_train = self.train_time * (self.max_train_count - self.train_count)
            s += f" {self.train_count} / {self.max_train_count}"
        else:
            remain_train = np.inf
        if self.timeout > 0:
            remain_time = self.timeout - elapsed_time
        else:
            remain_time = np.inf
        remain = min(remain_train, remain_time)
        s += f" {to_str_time(remain)}(remain time)"
        print(s)

        if len(self.progress_history) == 0:
            return

        info = self.progress_history[-1]
        train_count = info["train_count"]
        sync_count = info["sync_count"]
        remote_memory = info["remote_memory"]
        train_time = np.mean([t["train_time"] for t in self.progress_history])

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += " trainer:{:8d} tra".format(train_count)
        s += ",{:6.3f}s/tra".format(train_time)
        s += ",{:7d} memory ".format(remote_memory)
        s += ",{:6d} sync ".format(sync_count)

        if self.print_train_info:
            d = listdictdict_to_dictlist(self.progress_history, "train_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, actor_id, **kwargs):
        self.actor_id = actor_id
        if self.actor_id >= self.max_print_actor:
            return

        self.progress_t0 = time.time()
        self.progress_history = []

    def on_episodes_end(self, episode_count, **kwargs):
        if self.actor_id >= self.max_print_actor:
            return
        self._actor_print_progress(episode_count)

    def on_episode_begin(self, **kwargs):
        if self.actor_id >= self.max_print_actor:
            return
        self.history_step = []

    def on_step_end(
        self,
        env,
        workers,
        episode_count,
        step_time,
        **kwargs,
    ):
        if self.actor_id >= self.max_print_actor:
            return
        self.history_step.append(
            {
                "env_info": env.info,
                "work_info": workers[self.print_worker].info,
                "step_time": step_time,
            }
        )
        if self._check_print_progress():
            self._actor_print_progress(episode_count)

    def on_episode_end(
        self,
        episode_step,
        episode_count,
        episode_rewards,
        episode_time,
        valid_reward,
        worker_indices,
        **kwargs,
    ):
        if self.actor_id >= self.max_print_actor:
            return
        if len(self.history_step) == 0:
            return
        player_idx = worker_indices[self.print_worker]

        # 1エピソードの結果を平均でまとめる
        env_info = listdictdict_to_dictlist(self.history_step, "env_info")
        if "TimeLimit.truncated" in env_info:
            del env_info["TimeLimit.truncated"]
        for k, v in env_info.items():
            env_info[k] = np.mean(v)
        work_info = listdictdict_to_dictlist(self.history_step, "work_info")
        for k, v in work_info.items():
            work_info[k] = np.mean(v)

        epi_data = {
            "episode_count": episode_count,
            "episode_step": episode_step,
            "episode_reward": episode_rewards[player_idx],
            "episode_time": episode_time,
            "valid_reward": valid_reward,
            "step_time": np.mean([h["step_time"] for h in self.history_step]),
            "env_info": env_info,
            "work_info": work_info,
        }
        self.progress_history.append(epi_data)

    def _actor_print_progress(self, episode_count):

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f" actor{self.actor_id:2d}:"

        if len(self.progress_history) == 0:
            if len(self.history_step) > 0:
                step_num = len(self.history_step)
                step_time = np.mean([h["step_time"] for h in self.history_step])
                s += f" {episode_count:8d} episode"
                s += f", {step_num:5d} step"
                s += f", {step_time:.5f}s/step"
        else:
            episode_time = np.mean([h["episode_time"] for h in self.progress_history])

            s += " {:7d} epi".format(episode_count)
            s += f", {episode_time:.3f}s/epi"

            _r = [h["episode_reward"] for h in self.progress_history]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {min(_r):.3f} {np.mean(_r):.3f} {max(_r):.3f} reward"
            s += f", {np.mean(_s):.1f} step"

            valid_rewards = [t["valid_reward"] for t in self.progress_history if t["valid_reward"] is not None]
            if len(valid_rewards) > 0:
                s += ", {:.4f} val_reward ".format(np.mean(valid_rewards))

            if self.print_env_info:
                d = listdictdict_to_dictlist(self.progress_history, "env_info")
                for k, arr in d.items():
                    s += f"|{k} {np.mean(arr):.3f}"
            if self.print_worker_info:
                d = listdictdict_to_dictlist(self.progress_history, "work_info")
                for k, arr in d.items():
                    s += f"|{k} {np.mean(arr):.3f}"

        print(s)
        self.progress_history = []
