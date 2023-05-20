import io
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np

import srl
from srl.base.define import PlayRenderMode, RenderMode
from srl.base.env.base import EnvRun
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.utils.common import JsonNumpyEncoder, summarize_info_from_dictlist

logger = logging.getLogger(__name__)

"""
save_dir/
   ├ train_log/
   │ ├ actorX.txt
   │ ├ trainer.txt
   │ └ system.txt
   │
   ├ config.json
   ├ system.json
   └ version.txt

"""


@dataclass
class HistoryOnFile(Callback):
    save_dir: str
    log_interval: int = 1  # s

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[io.TextIOWrapper]] = {}

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"create dirs: {self.save_dir}")

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

    def _init(self, config: Config):
        # --- create dir
        self.param_dir = os.path.join(self.save_dir, "params")
        os.makedirs(self.param_dir, exist_ok=True)
        logger.info(f"create param_dir: {self.param_dir}")

        self.train_log_dir = os.path.join(self.save_dir, "train_log")
        os.makedirs(self.train_log_dir, exist_ok=True)
        logger.info(f"create train_log_dir: {self.train_log_dir}")

        # --- ver
        with open(os.path.join(self.save_dir, "version.txt"), "w", encoding="utf-8") as f:
            f.write(srl.__version__)

        # --- config
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        # --- system info
        if False:
            info = {}
            if config.enable_psutil:
                try:
                    import psutil

                    info["memory size"] = psutil.virtual_memory().total
                    info["memory percent"] = psutil.virtual_memory().percent
                    info["cpu count"] = os.cpu_count()
                    info["cpu(MHz)"] = [c.max for c in psutil.cpu_freq(percpu=True)]
                except Exception:
                    logger.info(traceback.format_exc())

            # GPU(nvidia) の計測をするかどうか
            if config.enable_nvidia:
                try:
                    import pynvml

                    info["nvidia driver version"] = str(pynvml.nvmlSystemGetDriverVersion())
                    info["gpu"] = []
                    for i in range(pynvml.nvmlDeviceGetCount()):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        info["gpu"].append(
                            {
                                "device": str(pynvml.nvmlDeviceGetName(handle)),
                                "memory": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                            }
                        )
                except Exception:
                    logger.info(traceback.format_exc())

            with open(os.path.join(self.save_dir, "system.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

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
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        self._init(info["config"])
        self.fp_dict["trainer"] = open(os.path.join(self.train_log_dir, "trainer.txt"), "w", encoding="utf-8")

        self.history: Dict[str, list] = {"train_time": []}
        self.t0 = time.time()
        self.interval_t0 = self.t0

    def on_trainer_end(self, info):
        self._write_trainer_log(info)
        self.close()

    def on_trainer_train(self, info):
        self.history["train_time"].append(info["train_time"])
        self._add_info(self.history, "trainer", info["train_info"])

        _time = time.time()
        if _time - self.interval_t0 > self.log_interval:
            self.interval_t0 = _time
            self._write_trainer_log(info)

    def _write_trainer_log(self, info):
        if self.fp_dict["trainer"] is None:
            return
        if len(self.history) == 0:
            return

        remote_memory = info["remote_memory"]
        _time = time.time()
        d = {
            "index": round((_time - self.t0) / self.log_interval),
            "time": _time - self.t0,
            "train": info["train_count"],
            "train_sync": info["sync"] if "sync" in info else 0,
            "remote_memory": 0 if remote_memory is None else remote_memory.length(),
        }
        d.update(summarize_info_from_dictlist(self.history))

        self._write_log(self.fp_dict["trainer"], d)
        self.history = {"train_time": []}

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, info):
        self._init(info["config"])

        self.player_num = info["env"].player_num
        self.actor_id = info["config"].actor_id

        path = os.path.join(self.train_log_dir, f"actor{self.actor_id}.txt")
        self.fp_dict["actor"] = open(path, "w", encoding="utf-8")
        if self.actor_id == 0:
            self.fp_dict["system"] = open(os.path.join(self.train_log_dir, "system.txt"), "w", encoding="utf-8")
        else:
            self.fp_dict["system"] = None

        self.last_episode_result = None
        self.t0 = time.time()
        self.interval_t0 = self.t0

    def on_episodes_end(self, info):
        self._write_actor_log(info)
        self.close()

    def on_episode_begin(self, info):
        self.step_infos = {}
        self._add_info(self.step_infos, "env", info["env"].info)

    def on_step_end(self, info):
        self._add_info(self.step_infos, "env", info["env"].info)
        self._add_info(self.step_infos, "trainer", info["train_info"])
        [self._add_info(self.step_infos, f"worker{i}", w.info) for i, w in enumerate(info["workers"])]

        _time = time.time()
        if _time - self.interval_t0 > self.log_interval:
            self.interval_t0 = _time
            self._write_actor_log(info)
            self._write_system_log(info["config"])

    def on_episode_end(self, info):
        d = summarize_info_from_dictlist(self.step_infos)
        d["episode_time"] = info["episode_time"]
        d["episode_step"] = info["episode_step"]
        for i, r in enumerate(info["episode_rewards"]):
            d[f"episode_reward{i}"] = r
        if info.get("eval_rewards", None) is not None:
            for i, r in enumerate(info["eval_rewards"]):
                d[f"eval_reward{i}"] = r
        self.last_episode_result = d

    def _write_actor_log(self, info):
        if self.fp_dict["actor"] is None:
            return
        if self.last_episode_result is None:
            return

        remote_memory = info["remote_memory"]
        trainer = info["trainer"]

        d = self.last_episode_result
        _time = time.time()
        d["index"] = round((_time - self.t0) / self.log_interval)
        d["time"] = _time - self.t0
        d["episode"] = info["episode_count"]
        d["sync"] = info["sync"] if "sync" in info else 0
        if trainer is not None:
            d["remote_memory"] = 0 if remote_memory is None else remote_memory.length()
            d["train"] = trainer.get_train_count()

        self._write_log(self.fp_dict["actor"], d)
        self.last_episode_result = None

    def _write_system_log(self, config: Config):
        if self.fp_dict["system"] is None:
            return

        _time = time.time()
        d: Dict[str, Any] = {
            "index": round((_time - self.t0) / self.log_interval),
            "time": _time - self.t0,
        }
        if config.enable_psutil:
            try:
                import psutil

                d["memory"] = psutil.virtual_memory().percent
                cpus = cast(List[float], psutil.cpu_percent(percpu=True))
                for i, cpu in enumerate(cpus):
                    d[f"cpu_{i}"] = cpu
                d["cpu"] = np.mean(cpus)
            except Exception:
                logger.debug(traceback.format_exc())

        if config.enable_nvidia:
            try:
                import pynvml

                gpu_num = pynvml.nvmlDeviceGetCount()
                if gpu_num > 0:
                    gpu = 0
                    gpu_memory = 0
                    for i in range(gpu_num):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        d[f"gpu_{i}"] = rate.gpu
                        d[f"gpu_{i}_memory"] = rate.memory
                        gpu += float(rate.gpu)
                        gpu_memory += float(rate.memory)
                    d["gpu"] = gpu / gpu_num
                    d["gpu_memory"] = gpu_memory / gpu_num
            except Exception:
                logger.debug(traceback.format_exc())

        self._write_log(self.fp_dict["system"], d)

    def _tmp_episode_env(self, info):
        if self.fp_dict["episode"] is None:
            return
        env: EnvRun = info["env"]
        d = {
            "step": env.step_num,
            "next_player_index": env.next_player_index,
            "state": env.state,
            "invalid_actions": env.get_invalid_actions(),
            "rewards": env.step_rewards,
            "done": env.done,
            "done_reason": env.done_reason,
            "time": info.get("step_time", 0),
            "env_info": env.info,
            "train_time": info.get("train_time", 0),
            "train_info": info.get("train_info", None),
        }
        if self.save_render:
            if "config" in info:
                config: Config = info["config"]
                render_mode = config.render_mode
                render_kwargs = config.render_kwargs
            else:
                render_mode = PlayRenderMode.rgb_array
                render_kwargs = {}
            render_mode = PlayRenderMode.convert_render_mode(render_mode)
            if render_mode == RenderMode.RBG_array:
                d["env_rgb_array"] = env.render_rgb_array(**render_kwargs)
            elif render_mode == RenderMode.Terminal:
                d["env_terminal"] = env.render_terminal(return_text=True, **render_kwargs)

        self.episode_info_env = d

    def _tmp_episode_worker(self, info):
        if self.fp_dict["episode"] is None:
            return
        d = {
            "action": info["action"],
        }
        if "workers" in info:
            workers: List[WorkerRun] = info["workers"]
            for i, w in enumerate(workers):
                d[f"work{i}_info"] = w.info
            if self.save_render:
                env: EnvRun = info["env"]
                config: Config = info["config"]
                for i, w in enumerate(workers):
                    render_mode = PlayRenderMode.convert_render_mode(config.render_mode)
                    if render_mode == RenderMode.RBG_array:
                        d[f"work{i}_rgb_array"] = w.render_rgb_array(env, **config.render_kwargs)
                    elif render_mode == RenderMode.Terminal:
                        d[f"work{i}_terminal"] = w.render_terminal(env, return_text=True, **config.render_kwargs)

        self.episode_info_worker = d

    def _write_episode_log(self):
        if self.fp_dict["episode"] is None:
            return
        d = {}
        d.update(self.episode_info_env)
        d.update(self.episode_info_worker)
        self._write_log(self.fp_dict["episode"], d)

    def _dict_update(self, d1, d2, prefix: str):
        if d2 is None:
            return
        for k, v in d2.items():
            key = f"{prefix}_{k}"
            if key in d1:
                key += "_2"
            d1[key] = v
