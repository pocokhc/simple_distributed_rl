import glob
import io
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

from srl.base.exception import UndefinedError
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RunContext
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callback import RunnerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.runner import Runner, RunnerConfig
from srl.utils.common import summarize_info_from_dictlist
from srl.utils.serialize import JsonNumpyEncoder

logger = logging.getLogger(__name__)

"""
save_dir/
   ├ actorX.txt
   ├ trainer.txt
   ├ system.txt
   ├ env_config.json
   ├ rl_config.json
   ├ context.json
   └ version.txt
"""


def _file_get_last_data(file_path) -> Optional[dict]:
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "rb") as f:
        cur = f.seek(-2, os.SEEK_END)
        while True:
            # --- 改行前まで進める
            while True:
                f.seek(cur, os.SEEK_SET)
                if f.read(1) == b"\n":
                    break
                cur -= 1
                if cur < 0:
                    break
            if cur < 0:
                break

            # --- data load
            line = f.readline().decode().strip()
            if line == "":
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSONDecodeError {e.args[0]}, '{line.strip()}'")

    return None


@dataclass
class HistoryOnFileBase:
    save_dir: str = "history"
    add_history: bool = False

    def __post_init__(self):
        self._fp_dict: dict[str, io.TextIOWrapper] = {}

    def __del__(self):
        self.close()

    def close(self):
        for k, v in self._fp_dict.items():
            if v is not None:
                try:
                    v.close()
                except Exception as e:
                    logger.error(f"close error: {e}")
        self._fp_dict = {}

    def open_fp(self, name: str, filename: str):
        if self.add_history:
            mode = "a"
        else:
            mode = "w"
        self._fp_dict[name] = open(os.path.join(self.save_dir, filename), mode, encoding="utf-8")

    def is_fp(self, name: str):
        return name in self._fp_dict

    def write_log(self, name: str, d):
        fp = self._fp_dict[name]
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        fp.flush()

    def setup(self, config: RunnerConfig, context: RunContext):
        import srl

        # --- make dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

        # --- ver
        path_ver = os.path.join(self.save_dir, "version.txt")
        with open(path_ver, "w", encoding="utf-8") as f:
            f.write(srl.__version__)

        # --- config
        for fn, dat in [
            ["env_config.json", context.env_config.to_dict()],
            ["rl_config.json", context.rl_config.to_dict()],
            ["context.json", context.to_dict(skip_config=True)],
            ["config.json", config.to_dict()],
        ]:
            path = os.path.join(self.save_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dat, f, indent=2)

        # --- 前回の終わり状況を読む
        self.start_time = 0
        self.start_episode_count = 0
        self.start_train_count = 0
        self.start_sync_actor = 0
        self.start_sync_trainer = 0
        if self.add_history:
            t_data = _file_get_last_data(os.path.join(self.save_dir, "trainer.txt"))
            if t_data is not None:
                self.start_time = t_data.get("time", 0)
                self.start_train_count = t_data.get("train", 0)
                self.start_sync_trainer = t_data.get("sync", 0)
            a_data = _file_get_last_data(os.path.join(self.save_dir, "actor0.txt"))
            if a_data is not None:
                if self.start_time < a_data.get("time", 0):
                    self.start_time = a_data.get("time", 0)
                if self.start_train_count < a_data.get("train", 0):
                    self.start_train_count = a_data.get("train", 0)
                self.start_episode_count = a_data.get("episode", 0)
                self.start_sync_actor = a_data.get("sync", 0)
        else:
            # 前回の結果を削除
            t_path = os.path.join(self.save_dir, "trainer.txt")
            if os.path.isfile(t_path):
                logger.info(f"remove file: {t_path}")
                os.remove(t_path)
            for path in glob.glob(os.path.join(self.save_dir, "actor*.txt")):
                logger.info(f"remove file: {path}")
                os.remove(path)
            s_path = os.path.join(self.save_dir, "system.txt")
            if os.path.isfile(s_path):
                logger.info(f"remove file: {s_path}")
                os.remove(s_path)


@dataclass
class HistoryOnFile(RunnerCallback, RunCallback, TrainerCallback, Evaluate):
    save_dir: str = "history"
    interval: int = 1
    interval_mode: str = "time"
    add_history: bool = False
    write_system: bool = False

    def __post_init__(self):
        self._base = HistoryOnFileBase(self.save_dir, self.add_history)
        assert self.interval_mode in ["time", "step"]

    def on_runner_start(self, runner: Runner) -> None:
        self._base.setup(runner.config, runner.context)

    def on_runner_end(self, runner: Runner) -> None:
        runner.history_viewer = HistoryViewer()
        runner.history_viewer.load(self.save_dir)

        # 2回目以降は引き継ぐ
        if runner._history_on_file_kwargs is not None:
            runner._history_on_file_kwargs["add_history"] = True

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
    # system
    # ---------------------------
    def _write_system(self):
        if not self._write_system:
            return
        if self.runner is None:
            return
        if not self.runner.config.enable_stats:
            return
        if not self._base.is_fp("system"):
            return

        d = {
            "name": "system",
            "time": time.time() - self.t0 + self._base.start_time,
        }

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

        self._base.write_log("system", d)

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        self._base.open_fp("actor", f"actor{context.actor_id}.txt")
        self._base.open_fp("system", "system.txt")

        # 分散の場合はactor_id=0のみevalをする
        if context.distributed:
            self.enable_eval = self.enable_eval and (context.actor_id == 0)

        self.t0 = time.time()
        if self.interval_mode == "time":
            self.interval0 = self.t0
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        self.last_episode_result = {}

    def on_episodes_end(self, context: RunContext, state: RunStateActor):
        self._write_actor_log(context, state)
        self._write_system()
        self._base.close()

    def on_episode_begin(self, context: RunContext, state: RunStateActor):
        self.episode_infos = {}
        self.t0_episode = time.time()
        self.step_count = 0

        if state.env is not None:
            self._add_info(self.episode_infos, "env", state.env.info)

    def on_step_end(self, context: RunContext, state: RunStateActor):
        self.step_count += 1

        if state.env is not None:
            self._add_info(self.episode_infos, "env", state.env.info)
        if state.trainer is not None:
            self._add_info(self.episode_infos, "trainer", state.trainer.train_info)
        [self._add_info(self.episode_infos, f"worker{i}", w.info) for i, w in enumerate(state.workers)]

        if self.interval_mode == "time":
            _time = time.time()
            if _time - self.interval0 > self.interval:
                self._write_actor_log(context, state)
                self._write_system()
                self.interval0 = _time  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 > self.interval:
                self._write_actor_log(context, state)
                self._write_system()
                self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

    def on_episode_end(self, context: RunContext, state: RunStateActor):
        d = {
            "episode": state.episode_count + self._base.start_episode_count,
            "episode_time": time.time() - self.t0_episode,
            "episode_step": self.step_count,
        }
        if state.env is not None:
            for i, r in enumerate(state.env.episode_rewards):
                d[f"reward{i}"] = r

        d.update(summarize_info_from_dictlist(self.episode_infos))

        self.last_episode_result = d

    def _write_actor_log(self, context: RunContext, state: RunStateActor):
        if not self._base.is_fp("actor"):
            return
        if self.last_episode_result is None:
            return

        d = self.last_episode_result
        d["name"] = f"actor{context.actor_id}"
        d["time"] = time.time() - self.t0 + self._base.start_time
        d["sync"] = state.sync_actor + self._base.start_sync_actor
        trainer = state.trainer
        if trainer is not None:
            d["train"] = trainer.get_train_count()
            memory = state.memory
            d["memory"] = 0 if memory is None else memory.length()
            if state.trainer is not None:
                for k, v in state.trainer.train_info.items():
                    d[f"trainer_{k}"] = v

        assert self.runner is not None
        if self.setup_eval_runner(self.runner):
            eval_rewards = self.run_eval(state.parameter)
            if eval_rewards is not None:
                for i, r in enumerate(eval_rewards):
                    d[f"eval_reward{i}"] = r

        self._base.write_log("actor", d)
        self.last_episode_result = None

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer):
        self._base.open_fp("trainer", "trainer.txt")
        self._base.open_fp("system", "system.txt")

        self.t0 = time.time()
        self.t0_train = time.time()
        self.t0_train_count = 0
        self.train_infos = {}

        if self.interval_mode == "time":
            self.interval0 = self.t0
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        # eval, 分散の場合はevalをしない
        if context.distributed:
            self.enable_eval = False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer):
        self._write_trainer_log(context, state)
        self._write_system()
        self._base.close()

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer):
        self._add_info(self.train_infos, "trainer", state.trainer.train_info)

        if self.interval_mode == "time":
            _time = time.time()
            if _time - self.interval0 > self.interval:
                self._write_trainer_log(context, state)
                self._write_system()
                self.interval0 = _time  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 > self.interval:
                self._write_trainer_log(context, state)
                self._write_system()
                self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

    def _write_trainer_log(self, context: RunContext, state: RunStateTrainer):
        if not self._base.is_fp("trainer"):
            return
        if self.train_infos == {}:
            return

        train_count = state.trainer.get_train_count()
        if train_count - self.t0_train_count > 0:
            train_time = (time.time() - self.t0_train) / (train_count - self.t0_train_count)
        else:
            train_time = np.inf
        d = {
            "name": "trainer",
            "time": time.time() - self.t0 + self._base.start_time,
            "train": train_count + self._base.start_train_count,
            "train_time": train_time,
            "sync": state.sync_trainer + self._base.start_sync_trainer,
        }
        memory = state.memory
        d["memory"] = 0 if memory is None else memory.length()

        assert self.runner is not None
        if self.setup_eval_runner(self.runner):
            eval_rewards = self.run_eval(state.parameter)
            if eval_rewards is not None:
                for i, r in enumerate(eval_rewards):
                    d[f"eval_reward{i}"] = r

        d.update(summarize_info_from_dictlist(self.train_infos))

        self._base.write_log("trainer", d)

        self.t0_train = time.time()
        self.t0_train_count = state.trainer.get_train_count()
        self.train_infos = {}
