import datetime as dt
import logging
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np

from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.utils.common import listdictdict_to_dictlist, summarize_info_from_dictlist, to_str_time

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
        if self.start_time > self.max_time:
            logger.info(f"change start_time: {self.start_time}s -> {self.max_time}s")
            self.start_time = self.max_time
        self.progress_timeout = self.start_time
        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

        self.resent_step_time: deque = deque(maxlen=10)
        self.resent_episode_time = deque(maxlen=10)
        self.resent_train_time = deque(maxlen=10)
        self.last_episode_count = 0
        self.last_train_count = 0
        self.last_memory = 0

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

    def on_episodes_begin(self, info):
        self.config: Config = info["config"]
        self.actor_id = info["actor_id"]

        if self.actor_id >= self.max_actor:
            return

        self.enable_ps = False
        try:
            if self.config.enable_ps:
                import psutil

                self._process = psutil.Process()
                self.enable_ps = True
        except Exception:
            logger.info(traceback.format_exc())

        self.enable_nvidia = self.config.enable_nvidia

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
        if self.actor_id >= self.max_actor:
            return

        self.last_episode_count = info["episode_count"]
        if info["trainer"] is not None:
            self.last_train_count = info["trainer"].get_train_count()
        self._print_actor()

    def on_episode_begin(self, info):
        if self.actor_id >= self.max_actor:
            return
        self.history_step = []
        self.last_episode_count = info["episode_count"]

    def on_step_end(self, info):
        if self.actor_id >= self.max_actor:
            return

        self.step_count += 1
        d = {
            "env_info": info["env"].info,
            "work_info": info["workers"][self.print_worker].info,
            "train_info": info["train_info"],
        }
        self.history_step.append(d)

        self.resent_step_time.append(info["step_time"])

        remote_memory = info["remote_memory"]
        self.last_memory = remote_memory.length() if remote_memory is not None else 0

        trainer = info["trainer"]
        if trainer is not None:
            self.last_train_count = trainer.get_train_count()
            self.resent_train_time.append(info["train_time"])

        if self._check_print_progress():
            self._print_actor()

    def on_episode_end(self, info):
        if self.actor_id >= self.max_actor:
            return

        self.resent_episode_time.append(info["episode_time"])

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
            "eval_reward": None if info.get("eval_rewards", None) is None else info["eval_rewards"][self.print_worker],
            "env_info": env_info,
            "work_info": work_info,
        }
        if "sync" in info:
            d["sync"] = info["sync"]

        # train info
        if self.history_step[0]["train_info"] is not None:
            train_info = listdictdict_to_dictlist(self.history_step, "train_info")
            d["train_info"] = summarize_info_from_dictlist(train_info)

        self.progress_history.append(d)

    # -----------------------------------------

    def _print_actor(self):
        elapsed_time = time.time() - self.t0

        # --- head
        # [TIME] [actor] [elapsed time]
        s = dt.datetime.now().strftime("%H:%M:%S")
        if self.config.distributed:
            s += f" actor{self.actor_id:2d}:"
        s += f" {to_str_time(elapsed_time)}"

        # [remain]
        step_time = np.mean(list(self.resent_step_time), dtype=float) if len(self.resent_step_time) > 0 else np.inf
        episode_time = (
            np.mean(list(self.resent_episode_time), dtype=float) if len(self.resent_episode_time) > 0 else np.inf
        )
        train_time = np.mean(list(self.resent_train_time), dtype=float) if len(self.resent_train_time) > 0 else np.inf
        if self.config.max_steps > 0:
            if len(self.resent_train_time) > 0 and train_time > 0:
                remain_step = (self.config.max_steps - self.step_count) * (step_time + train_time)
            else:
                remain_step = (self.config.max_steps - self.step_count) * step_time
        else:
            remain_step = np.inf
        if self.config.max_episodes > 0:
            remain_episode = (self.config.max_episodes - self.last_episode_count) * episode_time
        else:
            remain_episode = np.inf
        if self.config.timeout > 0:
            remain_time = self.config.timeout - elapsed_time
        else:
            remain_time = np.inf
        if self.config.max_train_count > 0:
            remain_train = (self.config.max_train_count - self.last_train_count) * train_time
        else:
            remain_train = np.inf
        remain = min(min(min(remain_step, remain_episode), remain_time), remain_train)
        if remain == np.inf:
            s += "(     - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [all step] [all episode] [train]
        s += f" {self.step_count:5d}st({self.last_episode_count:5d}ep)"
        if self.config.training and not self.config.distributed:
            s += " {:5d}tr".format(self.last_train_count)

        if len(self.progress_history) == 0:
            if len(self.history_step) == 0:
                # --- no info
                s += "1 step is not over."
            else:
                # --- steps info
                # [episode step] [step time]
                s += f", {len(self.history_step):5d} step"
                s += f", {step_time:.5f}s/step"

                # [train time]
                if self.config.training and not self.config.distributed:
                    s += f", {train_time:.5f}s/tr"

                # [memory_system]
                s += self._memory_system_str(self.last_memory)

                # [info]
                if self.print_env_info:
                    s += self._info_str(self.history_step, "env_info")
                if self.print_worker_info:
                    s += self._info_str(self.history_step, "work_info")
                if self.print_train_info:
                    s += self._info_str(self.history_step, "train_info")

        else:
            # --- episode info
            # [reward] [eval reward] [episode step] [episode time] [train time] [sync] [memory] [info]
            _r = [h["episode_reward"] for h in self.progress_history]
            s += f", {min(_r):.1f} {np.mean(_r):.3f} {max(_r):.1f} re"

            # [eval reward]
            s += self._eval_reward_str(self.progress_history)

            # [episode step] [episode time]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {np.mean(_s):.1f} step"
            s += f", {episode_time:.2f}s/ep"

            # [train time]
            if self.config.training and not self.config.distributed:
                s += f", {train_time:.3f}s/tr"

            # [sync]
            if "sync" in self.progress_history[0]:
                sync = max([h["sync"] for h in self.progress_history])
                s += f", {sync:3d} recv"

            # [memory_system]
            s += self._memory_system_str(self.last_memory)

            # [info]
            if self.print_env_info:
                s += self._info_str(self.progress_history, "env_info")
            if self.print_worker_info:
                s += self._info_str(self.progress_history, "work_info")
            if self.print_train_info:
                s += self._info_str(self.progress_history, "train_info")

        print(s)
        self.progress_history = []

    def _eval_reward_str(self, arr) -> str:
        if arr is None or len(arr) == 0:
            if self.config.distributed:
                return " " * 12
            else:
                return ""
        _rewards = [h["eval_reward"] for h in arr if h["eval_reward"] is not None]
        if len(_rewards) > 0:
            s = f"({np.mean(_rewards):.3f} eval)"
        elif self.config.distributed:
            s = " " * 12
        else:
            s = ""
        return s

    def _memory_system_str(self, memory_len) -> str:
        # , 1234567mem,CPU100% M100%,GPU0 100% M100%
        s = ""
        if self.actor_id != 0:
            if self.enable_ps:
                try:
                    import psutil

                    cpu_percent = self._process.cpu_percent(None) / psutil.cpu_count()
                    s += f"[CPU{cpu_percent:3.0f}%]"

                except Exception:
                    logger.debug(traceback.format_exc())
                    s += "[CPU Nan%]"

        else:
            s = f", {memory_len:5d}mem"

            if self.enable_ps:
                try:
                    import psutil

                    # CPU,memory
                    memory_percent = psutil.virtual_memory().percent
                    cpu_percent = self._process.cpu_percent(None) / psutil.cpu_count()
                    s += f"[CPU{cpu_percent:3.0f}%,M{memory_percent:2.0f}%]"

                except Exception:
                    logger.debug(traceback.format_exc())
                    s += "[CPU Nan%]"

            if self.enable_nvidia:
                try:
                    import pynvml

                    gpu_num = pynvml.nvmlDeviceGetCount()
                    gpus = []
                    for device_id in range(gpu_num):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpus.append(f"[GPU{device_id} {rate.gpu:2.0f}%,M{rate.memory:2.0f}%]")
                    s += "".join(gpus)

                except Exception:
                    logger.debug(traceback.format_exc())
                    s += ",GPU Nan%"

        return s

    def _info_str(self, arr_dict, key) -> str:
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

    # ----------------------------------
    # trainer
    # ----------------------------------
    def on_trainer_start(self, info) -> None:
        self.config: Config = info["config"]
        remote_memory = info["remote_memory"]

        if not self.config.distributed:
            print(
                "### max train: {}, timeout: {}, memory len: {}".format(
                    self.config.max_train_count,
                    to_str_time(self.config.timeout),
                    remote_memory.length(),
                )
            )

        self.enable_ps = False
        try:
            if self.config.enable_ps:
                import psutil

                self._process = psutil.Process()
                self.enable_ps = True
        except Exception:
            logger.info(traceback.format_exc())

        self.enable_nvidia = self.config.enable_nvidia

        self.progress_t0 = self.t0 = time.time()
        self.progress_history = []

        self.resent_train_time = deque(maxlen=10)
        self.last_train_count = 0

    def on_trainer_end(self, info) -> None:
        self.last_train_count: int = info["train_count"]
        self._print_trainer()

    def on_trainer_train(self, info) -> None:
        self.resent_train_time.append(info["train_time"])
        self.last_train_count: int = info["train_count"]

        d = {
            "train_info": info["train_info"],
        }
        if info.get("eval_rewards", None) is not None:
            d["eval_reward"] = info["eval_rewards"][self.print_worker]
        if "sync" in info:
            d["sync"] = info["sync"]

        self.progress_history.append(d)
        if self._check_print_progress():
            self._print_trainer()

    def _print_trainer(self):
        elapsed_time = time.time() - self.t0

        # --- head
        # [TIME] [trainer] [elapsed time] [train count]
        s = dt.datetime.now().strftime("%H:%M:%S")
        if self.config.distributed:
            s += " trainer:"
        s += f" {to_str_time(elapsed_time)}"

        # [remain]
        train_time = np.mean(list(self.resent_train_time), dtype=float) if len(self.resent_train_time) > 0 else np.inf
        if self.config.max_train_count > 0 and len(self.resent_train_time) > 0 and train_time > 0:
            remain_train = (self.config.max_train_count - self.last_train_count) * train_time
        else:
            remain_train = np.inf
        if self.config.timeout > 0:
            remain_time = self.config.timeout - elapsed_time
        else:
            remain_time = np.inf
        remain = min(remain_train, remain_time)
        if remain == np.inf:
            s += "(      - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [train count]
        s += " {:6d}tr".format(self.last_train_count)

        if len(self.progress_history) == 0:
            # --- no info
            s += "1 train is not over."
        else:
            # --- train info
            # [train time] [eval] [sync] [info]
            s += f", {train_time:.4f}s/tr"

            # [eval]
            if "eval_reward" in self.progress_history[0]:
                eval_rewards = [h["eval_reward"] for h in self.progress_history]
                if len(eval_rewards) > 0:
                    s += f", {np.mean(eval_rewards):.3f} eval reward"
            # [sync]
            if "sync" in self.progress_history[0]:
                sync = max([h["sync"] for h in self.progress_history])
                s += f", {sync:3d} send "

            # [system]
            if self.config.distributed:
                if self.enable_ps:
                    try:
                        import psutil

                        cpu_percent = self._process.cpu_percent(None) / psutil.cpu_count()
                        s += f"[CPU{cpu_percent:3.0f}%]"

                    except Exception:
                        logger.debug(traceback.format_exc())
                        s += "[CPU Nan%]"

            else:
                if self.enable_ps:
                    try:
                        import psutil

                        # CPU,memory
                        memory_percent = psutil.virtual_memory().percent
                        cpu_percent = self._process.cpu_percent(None) / psutil.cpu_count()
                        s += f"[CPU{cpu_percent:3.0f}%,M{memory_percent:2.0f}%]"

                    except Exception:
                        logger.debug(traceback.format_exc())
                        s += "[CPU Nan%]"

                if self.enable_nvidia:
                    try:
                        import pynvml

                        gpu_num = pynvml.nvmlDeviceGetCount()
                        gpus = []
                        for device_id in range(gpu_num):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                            rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpus.append(f"[GPU{device_id} {rate.gpu:2.0f}%,M{rate.memory:2.0f}%]")
                        s += "".join(gpus)

                    except Exception:
                        logger.debug(traceback.format_exc())
                        s += ",GPU Nan%"

            # [train info]
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

    # ----------------------------------
    # mp
    # ----------------------------------
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
