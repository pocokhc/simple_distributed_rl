import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np

from srl.runner.callback import Callback, TrainerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner
from srl.utils.common import summarize_info_from_dictlist

logger = logging.getLogger(__name__)


@dataclass
class HistoryOnMemory(Callback, TrainerCallback, Evaluate):
    """memory上に保存する、distributeでは実行しない"""

    interval: int = 1  # s

    def _read_stats(self, runner: Runner):
        if not runner.config.enable_stats:
            return {}

        d = {}

        if runner.config.used_psutil:
            try:
                memory_percent, cpu_percent = runner.read_psutil()
                d["system_memory"] = memory_percent
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

    def on_episodes_begin(self, runner: Runner):
        assert not runner.context.distributed

        self.t0 = time.time()
        self.setup_eval_runner(runner)

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

    def on_episode_end(self, runner: Runner):
        d = {
            "name": f"actor{runner.context.actor_id}",
            "time": time.time() - self.t0,
            "episode": runner.state.episode_count,
            "episode_time": time.time() - self.t0_episode,
            "episode_step": self.step_count,
            "sync": runner.state.sync_actor,
        }
        if runner.state.env is not None:
            for i, r in enumerate(runner.state.env.episode_rewards):
                d[f"reward{i}"] = r

        trainer = runner.state.trainer
        if trainer is not None:
            d["train"] = trainer.get_train_count()
            memory = runner.state.memory
            d["memory"] = 0 if memory is None else memory.length()
            if runner.state.trainer is not None:
                for k, v in runner.state.trainer.train_info.items():
                    d[f"trainer_{k}"] = v

        if self.enable_eval:
            eval_rewards = self.run_eval(runner.state.parameter)
            for i, r in enumerate(eval_rewards):
                d[f"eval_reward{i}"] = r

        d.update(summarize_info_from_dictlist(self.episode_infos))
        d.update(self._read_stats(runner))

        runner._history.append(d)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, runner: Runner):
        self.t0 = time.time()
        self.t0_train = time.time()
        self.t0_train_count = 0
        self.interval_t0 = time.time()
        self.train_infos = {}

    def on_trainer_end(self, runner: Runner):
        self._save_trainer_log(runner)

    def on_trainer_loop(self, runner: Runner):
        assert runner.state.trainer is not None
        self._add_info(self.train_infos, "trainer", runner.state.trainer.train_info)

        _time = time.time()
        if _time - self.interval_t0 > self.interval:
            self._save_trainer_log(runner)
            self.interval_t0 = _time

    def _save_trainer_log(self, runner: Runner):
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
        d.update(self._read_stats(runner))

        runner._history.append(d)

        self.t0_train = time.time()
        self.t0_train_count = runner.state.trainer.get_train_count()
        self.train_infos = {}
