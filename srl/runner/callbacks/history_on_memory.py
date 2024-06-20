import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np

from srl.base.context import RunContext
from srl.base.exception import UndefinedError
from srl.base.run.callback import RunCallback, TrainCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callback import RunnerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class HistoryOnMemory(RunCallback, TrainCallback, Evaluate):
    interval: int = 1
    interval_mode: str = "step"

    def on_runner_end(self, runner: Runner) -> None:
        runner.history_viewer = HistoryViewer()
        runner.history_viewer.set_history_on_memory(self, runner)

    def _read_stats(self):
        assert self.runner is not None
        if not self.runner.config.enable_stats:
            return {}

        d = {}

        try:
            memory_percent, cpu_percent = self.runner.read_psutil()
            if memory_percent != np.NaN:
                d["system_memory"] = memory_percent
                d["cpu"] = cpu_percent
        except Exception:
            logger.debug(traceback.format_exc())

        try:
            gpus = self.runner.read_nvml()
            # device_id, rate.gpu, rate.memory
            for device_id, gpu, memory in gpus:
                d[f"gpu{device_id}"] = gpu
                d[f"gpu{device_id}_memory"] = memory
        except Exception:
            logger.debug(traceback.format_exc())

        return d

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        assert not context.distributed, "Not supported in distributed."
        self.logs = []
        self.t0 = time.time()

        if self.interval_mode == "time":
            self.interval0 = self.t0
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        self._is_immediately_after_writing = False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs):
        if not self._is_immediately_after_writing:
            self._append_infos_for_actor(context, state)

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs):
        self._is_immediately_after_writing = False
        if self.interval_mode == "time":
            if time.time() - self.interval0 > self.interval:
                self._append_infos_for_actor(context, state)
                self._is_immediately_after_writing = True
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._append_infos_for_actor(context, state)
                self._is_immediately_after_writing = True
                self.interval0 = 0  # last
        else:
            raise UndefinedError(self.interval_mode)

    def _append_infos_for_actor(self, context: RunContext, state: RunStateActor):
        d = {
            "name": f"actor{context.actor_id}",
            "time": time.time() - self.t0,
            "step": state.total_step,
            "episode": state.episode_count,
            "episode_time": state.last_episode_time,
            "episode_step": state.last_episode_step,
            "sync": state.sync_actor,
        }
        for i, r in enumerate(state.last_episode_rewards):
            d[f"reward{i}"] = r

        # --- env
        if state.env is not None:
            for k, v in state.env.info.items():
                d[f"env_{k}"] = v

        # --- workers
        for i, w in enumerate(state.workers):
            for k, v in w.info.items():
                d[f"worker{i}_{k}"] = v

        # --- trainer
        if state.trainer is not None:
            d["train"] = state.trainer.get_train_count()
            d["memory"] = 0 if state.memory is None else state.memory.length()
            for k, v in state.trainer.info.items():
                d[f"trainer_{k}"] = v

        # --- system
        d.update(self._read_stats())

        # --- eval
        assert self.runner is not None
        if self.setup_eval_runner(self.runner):
            eval_rewards = self.run_eval(state.parameter)
            if eval_rewards is not None:
                for i, r in enumerate(eval_rewards):
                    d[f"eval_reward{i}"] = r

        self.logs.append(d)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer, **kwargs):
        assert not context.distributed, "Not supported in distributed."
        self.logs = []
        self.t0 = time.time()

        self.t0_train = time.time()
        self.t0_train_count = 0

        if self.interval_mode == "time":
            self.interval0 = self.t0
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        self._is_immediately_after_writing = False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs):
        if not self._is_immediately_after_writing:
            self._append_infos_for_trainer(context, state)

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        self._is_immediately_after_writing = False
        if self.interval_mode == "time":
            if time.time() - self.interval0 > self.interval:
                self._append_infos_for_trainer(context, state)
                self._is_immediately_after_writing = True
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._append_infos_for_trainer(context, state)
                self._is_immediately_after_writing = True
                self.interval0 = 0  # last
        else:
            raise UndefinedError(self.interval_mode)
        return False

    def _append_infos_for_trainer(self, context: RunContext, state: RunStateTrainer):
        # --- calc train time
        train_count = state.trainer.get_train_count()
        if train_count - self.t0_train_count > 0:
            train_time = (time.time() - self.t0_train) / (train_count - self.t0_train_count)
        else:
            train_time = np.inf
        self.t0_train = time.time()
        self.t0_train_count = state.trainer.get_train_count()

        d = {
            "name": "trainer",
            "time": time.time() - self.t0,
            "train": train_count,
            "train_time": train_time,
            "sync": state.sync_trainer,
        }
        d["memory"] = 0 if state.memory is None else state.memory.length()

        # --- trainer
        for k, v in state.trainer.info.items():
            d[f"trainer_{k}"] = v

        # --- system
        d.update(self._read_stats())

        self.logs.append(d)
