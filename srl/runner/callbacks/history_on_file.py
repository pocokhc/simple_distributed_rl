import io
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

import srl
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner
from srl.utils.common import summarize_info_from_dictlist
from srl.utils.serialize import JsonNumpyEncoder

logger = logging.getLogger(__name__)

"""
save_dir/
   ├ log/
   │ ├ actorX.txt
   │ └ trainer.txt
   │
   ├ env_config.json
   ├ rl_config.json
   ├ context.json
   └ version.txt
"""


@dataclass
class HistoryOnFile(Callback, TrainerCallback, Evaluate):
    interval: int = 1  # s

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

        # --- file
        path = os.path.join(self.save_dir, "env_config.json")
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(runner.env_config.to_dict(), f, indent=2)

        path = os.path.join(self.save_dir, "rl_config.json")
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(runner.rl_config.to_dict(), f, indent=2)

        path = os.path.join(self.save_dir, "context.json")
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(runner.context.to_dict(), f, indent=2)

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
        self.create_eval_runner(runner)

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
        if runner.state.trainer is not None:
            self._add_info(self.episode_infos, "trainer", runner.state.trainer.train_info)
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
            memory = runner.state.memory
            d["memory"] = 0 if memory is None else memory.length()
            if runner.state.trainer is not None:
                for k, v in runner.state.trainer.train_info.items():
                    d[f"trainer_{k}"] = v

        eval_rewards = self.run_eval(runner)
        if eval_rewards is not None:
            for i, r in enumerate(eval_rewards):
                d[f"eval_reward{i}"] = r

        d.update(self._read_stats(runner))

        self._write_log(self.fp_dict["actor"], d)
        self.last_episode_result = None

    def _read_stats(self, runner: Runner):
        if not runner.config.enable_stats:
            return {}

        d = {}

        if runner.config.used_psutil:
            try:
                memory_percent, cpu_percent = runner.read_psutil()
                d["memory"] = memory_percent
                d["cpu"] = cpu_percent
            except Exception:
                logger.debug(traceback.format_exc())

        if runner.config.used_nvidia:
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

    def on_trainer_train_end(self, runner: Runner):
        assert runner.state.trainer is not None
        self._add_info(self.train_infos, "trainer", runner.state.trainer.train_info)

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
        memory = runner.state.memory
        d["memory"] = 0 if memory is None else memory.length()

        d.update(summarize_info_from_dictlist(self.train_infos))
        # d.update(self._read_stats(runner))

        self._write_log(self.fp_dict["trainer"], d)

        self.t0_train = time.time()
        self.t0_train_count = runner.state.trainer.get_train_count()
        self.train_infos = {}
