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

from srl.base.context import RunContext
from srl.base.exception import UndefinedError
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.evaluate import Evaluate
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
    save_dir: str
    add_history: bool

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

    def setup(self, context: RunContext):
        import srl

        if self.save_dir == "":
            import tempfile

            self.save_dir = tempfile.mkdtemp()
            logger.info(f"tempfile.mkdtemp(): {self.save_dir}")
        elif not os.path.isdir(self.save_dir):
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
            ["context.json", context.to_dict()],
        ]:
            path = os.path.join(self.save_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dat, f, indent=2)

        # --- 前回の終わり状況を読む
        self.start_time = 0
        self.start_total_step = 0
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
                self.start_total_step = a_data.get("step", 0)
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
class HistoryOnFile(RunCallback, Evaluate):
    save_dir: str = ""
    interval: int = 1
    interval_mode: str = "time"
    add_history: bool = False
    write_system: bool = False

    def __post_init__(self):
        self._base = HistoryOnFileBase(self.save_dir, self.add_history)
        assert self.interval_mode in ["time", "step"]

    def on_start(self, context: RunContext, **kwargs) -> None:
        self._base.setup(context)

    # ---------------------------
    # system
    # ---------------------------
    def _write_system(self, context: RunContext):
        if not self._write_system:
            return
        if not context.enable_stats:
            return
        if not self._base.is_fp("system"):
            return

        d = {
            "name": "system",
            "time": time.time() - self.t0 + self._base.start_time,
        }

        try:
            from srl.base.system import psutil_

            d["system_memory"] = psutil_.read_memory()
            d["cpu"] = psutil_.read_cpu()
        except Exception:
            logger.debug(traceback.format_exc())

        try:
            from srl.base.system.pynvml_ import read_nvml

            gpus = read_nvml()
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
    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
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

        self._is_immediately_after_writing = False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs):
        if not self._is_immediately_after_writing:
            self._write_actor_log(context, state)
            self._write_system(context)
        self._base.close()

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs):
        self._is_immediately_after_writing = False
        if self.interval_mode == "time":
            if time.time() - self.interval0 > self.interval:
                self._write_actor_log(context, state)
                self._write_system(context)
                self._is_immediately_after_writing = True
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._write_actor_log(context, state)
                self._write_system(context)
                self._is_immediately_after_writing = True
                self.interval0 = 0  # last
        else:
            raise UndefinedError(self.interval_mode)

    def _write_actor_log(self, context: RunContext, state: RunStateActor):
        if not self._base.is_fp("actor"):
            return

        d = {}
        d["name"] = f"actor{context.actor_id}"
        d["time"] = time.time() - self.t0 + self._base.start_time
        d["sync"] = state.sync_actor + self._base.start_sync_actor
        d["step"] = state.total_step + self._base.start_total_step
        d["episode"] = state.episode_count + self._base.start_episode_count
        d["episode_time"] = state.last_episode_time
        d["episode_step"] = state.last_episode_step
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

        # --- eval
        eval_rewards = self.run_eval_with_state(context, state)
        if eval_rewards is not None:
            for i, r in enumerate(eval_rewards):
                d[f"eval_reward{i}"] = r

        self._base.write_log("actor", d)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer, **kwargs):
        self._base.open_fp("trainer", "trainer.txt")
        self._base.open_fp("system", "system.txt")
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

        # eval, 分散の場合はevalをしない
        if context.distributed:
            self.enable_eval = False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs):
        if not self._is_immediately_after_writing:
            self._write_trainer_log(context, state)
            self._write_system(context)
        self._base.close()

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        self._is_immediately_after_writing = False
        if self.interval_mode == "time":
            if time.time() - self.interval0 > self.interval:
                self._write_trainer_log(context, state)
                self._write_system(context)
                self._is_immediately_after_writing = True
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._write_trainer_log(context, state)
                self._write_system(context)
                self._is_immediately_after_writing = True
                self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)
        return False

    def _write_trainer_log(self, context: RunContext, state: RunStateTrainer):
        if not self._base.is_fp("trainer"):
            return
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
            "time": time.time() - self.t0 + self._base.start_time,
            "train": train_count + self._base.start_train_count,
            "train_time": train_time,
            "sync": state.sync_trainer + self._base.start_sync_trainer,
        }
        d["memory"] = 0 if state.memory is None else state.memory.length()

        # --- trainer
        for k, v in state.trainer.info.items():
            d[f"trainer_{k}"] = v

        # --- eval
        eval_rewards = self.run_eval_with_state(context, state)
        if eval_rewards is not None:
            for i, r in enumerate(eval_rewards):
                d[f"eval_reward{i}"] = r

        self._base.write_log("trainer", d)
