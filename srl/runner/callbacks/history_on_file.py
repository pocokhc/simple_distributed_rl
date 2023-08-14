import io
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

import srl
from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback, CallbackType, TrainerCallback
from srl.runner.runner import Runner
from srl.utils.common import JsonNumpyEncoder, summarize_info_from_dictlist

logger = logging.getLogger(__name__)

"""
save_dir/
   ├ log/
   │ ├ actorX.txt
   │ └ trainer.txt
   │
   ├ config.json
   ├ context.json
   └ version.txt
"""


@dataclass
class HistoryOnFile(Callback, TrainerCallback):
    interval: int = 1  # s

    enable_eval: bool = True
    eval_env_sharing: bool = False
    eval_episode: int = 10
    eval_timeout: int = -1
    eval_max_steps: int = -1
    eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    eval_shuffle_player: bool = False
    eval_used_device_tf: str = "/CPU"
    eval_used_device_torch: str = "cpu"
    eval_callbacks: List[CallbackType] = field(default_factory=list)

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[io.TextIOWrapper]] = {}

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self.fp_dict.items():
            if v is not None:
                self.fp_dict[k] = None
                v.close()

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    def _init(self, runner: Runner):
        self.save_dir = runner.context.save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        self.log_dir = os.path.join(self.save_dir, "logs")
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"create log_dir: {self.log_dir}")

        # --- ver
        path_ver = os.path.join(self.save_dir, "version.txt")
        if not os.path.isfile(path_ver):
            with open(path_ver, "w", encoding="utf-8") as f:
                f.write(srl.__version__)

        # --- config
        path_conf = os.path.join(self.save_dir, "config.json")
        if not os.path.isfile(path_conf):
            with open(path_conf, "w", encoding="utf-8") as f:
                json.dump(runner.config.to_json_dict(), f, indent=2)

        path_cont = os.path.join(self.save_dir, "context.json")
        if not os.path.isfile(path_cont):
            with open(path_cont, "w", encoding="utf-8") as f:
                json.dump(runner.context.to_json_dict(), f, indent=2)

    def _add_info(self, info, prefix, dict_):
        if dict_ is None:
            return
        for k, v in dict_.items():
            k = f"{prefix}_{k}"
            if k not in info:
                info[k] = [v]
            else:
                info[k].append(v)

    # ---------------------------
    # eval
    # ---------------------------
    def _create_eval_runner(self, runner: Runner):
        if not self.enable_eval:
            return
        self.eval_runner = runner.create_eval_runner(self.eval_env_sharing)

        # config
        self.eval_runner.config.players = self.eval_players
        self.eval_runner.config.seed = None
        self.eval_runner.context.used_device_tf = self.eval_used_device_tf
        self.eval_runner.context.used_device_torch = self.eval_used_device_torch

        # context
        self.eval_runner.context.max_episodes = self.eval_episode
        self.eval_runner.context.timeout = self.eval_timeout
        self.eval_runner.context.max_steps = self.eval_max_steps
        self.eval_runner.context.shuffle_player = self.eval_shuffle_player
        self.eval_runner.context.callbacks = self.eval_callbacks
        self.eval_runner.context.init(self.eval_runner)

    def _eval(self, runner: Runner) -> str:
        if not self.enable_eval:
            return ""
        self.eval_runner._play(runner.parameter, runner.remote_memory)
        eval_rewards = self.eval_runner.state.episode_rewards_list
        eval_rewards = np.mean(eval_rewards, axis=0)
        return eval_rewards

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, runner: Runner):
        self._init(runner)

        self.actor_id = runner.context.actor_id

        path = os.path.join(self.log_dir, f"actor{self.actor_id}.txt")
        self.fp_dict["actor"] = open(path, "w", encoding="utf-8")

        self.t0 = time.time()
        self.interval_t0 = self.t0
        self.last_episode_result = {}
        self._create_eval_runner(runner)

    def on_episodes_end(self, runner: Runner):
        self._write_actor_log(runner)
        self.close()

    def on_episode_begin(self, runner: Runner):
        self.episode_infos = {}
        self.t0_episode = time.time()
        self.step_count = 0

        if runner.state.env is not None:
            self._add_info(self.episode_infos, "env", runner.state.env.info)

    def on_step_end(self, runner: Runner):
        self.step_count += 1

        if runner.state.env is not None:
            self._add_info(self.episode_infos, "env", runner.state.env.info)
        self._add_info(self.episode_infos, "trainer", runner.state.train_info)
        [self._add_info(self.episode_infos, f"worker{i}", w.info) for i, w in enumerate(runner.state.workers)]

        _time = time.time()
        if _time - self.interval_t0 > self.interval:
            self.interval_t0 = _time
            self._write_actor_log(runner)

    def on_episode_end(self, runner: Runner):
        d = {
            "episode": runner.state.episode_count,
            "episode_time": time.time() - self.t0_episode,
            "episode_step": self.step_count,
        }
        if runner.state.env is not None:
            for i, r in enumerate(runner.state.env.episode_rewards):
                d[f"reward{i}"] = r

        d.update(summarize_info_from_dictlist(self.episode_infos))

        self.last_episode_result = d

    def _write_actor_log(self, runner: Runner):
        if self.fp_dict["actor"] is None:
            return
        if self.last_episode_result is None:
            return

        d = self.last_episode_result
        _time = time.time()
        d["name"] = f"actor{runner.context.actor_id}"
        d["time"] = _time - self.t0
        d["sync"] = runner.state.sync_actor
        trainer = runner.state.trainer
        if trainer is not None:
            d["train"] = trainer.get_train_count()
            remote_memory = runner.state.remote_memory
            d["remote_memory"] = 0 if remote_memory is None else remote_memory.length()
            if runner.state.train_info is not None:
                for k, v in runner.state.train_info.items():
                    d[f"trainer_{k}"] = v

        if self.enable_eval:
            eval_rewards = self._eval(runner)
            for i, r in enumerate(eval_rewards):
                d[f"eval_reward{i}"] = r

        d.update(self._read_stats(runner))

        self._write_log(self.fp_dict["actor"], d)
        self.last_episode_result = None

    def _read_stats(self, runner: Runner):
        if not runner.config.enable_stats:
            return {}

        d = {}

        if runner.context.used_psutil:
            try:
                memory_percent, cpu_percent = runner.read_psutil()
                d["memory"] = memory_percent
                d["cpu"] = cpu_percent
            except Exception:
                logger.debug(traceback.format_exc())

        if runner.context.used_nvidia:
            try:
                gpus = runner.read_nvml()
                # device_id, rate.gpu, rate.memory
                for device_id, gpu, memory in gpus:
                    d[f"gpu{device_id}"] = gpu
                    d[f"gpu{device_id}_memory"] = memory
            except Exception:
                logger.debug(traceback.format_exc())

        return d

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, runner: Runner):
        self._init(runner)
        self.fp_dict["trainer"] = open(os.path.join(self.log_dir, "trainer.txt"), "w", encoding="utf-8")

        self.t0 = time.time()
        self.t0_train = time.time()
        self.t0_train_count = 0
        self.interval_t0 = time.time()
        self.train_infos = {}

    def on_trainer_end(self, runner: Runner):
        self._write_trainer_log(runner)
        self.close()

    def on_trainer_train(self, runner: Runner):
        assert runner.state.trainer is not None
        self._add_info(self.train_infos, "trainer", runner.state.train_info)

        _time = time.time()
        if _time - self.interval_t0 > self.interval:
            self._write_trainer_log(runner)
            self.interval_t0 = _time

    def _write_trainer_log(self, runner: Runner):
        if self.fp_dict["trainer"] is None:
            return
        if self.train_infos == {}:
            return
        assert runner.state.trainer is not None

        train_count = runner.state.trainer.get_train_count()
        if train_count - self.t0_train_count > 0:
            train_time = (time.time() - self.t0_train) / (train_count - self.t0_train_count)
        else:
            train_time = np.inf
        d = {
            "name": "trainer",
            "time": time.time() - self.t0,
            "train": train_count,
            "train_time": train_time,
            "sync": runner.state.sync_trainer,
        }
        remote_memory = runner.state.remote_memory
        d["remote_memory"] = 0 if remote_memory is None else remote_memory.length()

        d.update(summarize_info_from_dictlist(self.train_infos))
        # d.update(self._read_stats(runner))

        self._write_log(self.fp_dict["trainer"], d)

        self.t0_train = time.time()
        self.t0_train_count = runner.state.trainer.get_train_count()
        self.train_infos = {}
