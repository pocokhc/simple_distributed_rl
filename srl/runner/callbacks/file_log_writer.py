import datetime as dt
import io
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np
import srl
from srl.base.define import PlayRenderMode, RenderMode
from srl.base.env.base import EnvRun
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback, GameCallback
from srl.runner.config import Config
from srl.utils.common import JsonNumpyEncoder, summarize_info_from_list

logger = logging.getLogger(__name__)

"""
tmp/
 └ YYYYMMDD_HHMMSS_EnvName_RLName/
   ├ train_log/
   │ ├ actorX.txt
   │ ├ trainer.txt
   │ └ system.txt
   │
   ├ episode_log/
   │ └ episodeX.txt
   │
   ├ params/
   │ └ xxxxx.pickle
   │
   ├ config.json
   ├ system.json
   └ version.txt

"""


@dataclass
class FileLogWriter(Callback, GameCallback):
    tmp_dir: str = "tmp"

    # train log
    enable_train_log: bool = True
    train_log_interval: int = 1  # s

    # episode log
    enable_episode_log: bool = False
    add_render: bool = True

    # checkpoint
    enable_checkpoint: bool = True
    checkpoint_interval: int = 60 * 20  # s

    def __post_init__(self):
        self.fp_dict: dict[str, Optional[io.TextIOWrapper]] = {}
        self.base_dir = ""

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

    def on_init(self, info) -> None:
        self._init_dir(info["config"])

    def _init_dir(self, config: Config):
        if self.base_dir != "":
            return

        dir_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{config.env_config.name}_{config.rl_config.getName()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        self.base_dir = os.path.join(os.path.abspath(self.tmp_dir), dir_name)
        logger.debug(f"save path: {self.base_dir}")

        self.param_dir = os.path.join(self.base_dir, "params")
        os.makedirs(self.param_dir, exist_ok=True)

        self.train_log_dir = os.path.join(self.base_dir, "train_log")
        os.makedirs(self.train_log_dir, exist_ok=True)

        self.episode_log_dir = os.path.join(self.base_dir, "episode_log")
        os.makedirs(self.episode_log_dir, exist_ok=True)

        # ver
        with open(os.path.join(self.base_dir, "version.txt"), "w", encoding="utf-8") as f:
            f.write(srl.__version__)

    def _write_config_summary(self, config):
        assert self.base_dir != ""
        with open(os.path.join(self.base_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    def _write_system_info_summary(self, config: Config):
        assert self.base_dir != ""
        info = {}
        if config.enable_ps:
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

        with open(os.path.join(self.base_dir, "system.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        config = info["config"]
        self._init_dir(config)
        self._write_config_summary(config)
        self.fp_dict["trainer"] = open(os.path.join(self.train_log_dir, "trainer.txt"), "w", encoding="utf-8")

        _time = time.time()

        # train logger
        if self.enable_train_log:
            self.log_history = []
            self.log_t0 = _time

        # checkpoint
        if self.enable_checkpoint:
            self.checkpoint_t0 = _time
            self.env = config.make_env()

    def on_trainer_end(self, info):
        if self.enable_train_log:
            self._write_trainer_log()
        if self.enable_checkpoint:
            self._save_parameter(info["parameter"], info["train_count"])

        self.close()

    def on_trainer_train(self, info):
        _time = time.time()

        if self.enable_train_log:
            remote_memory = info["remote_memory"]

            d = {
                "train_count": info["train_count"],
                "train_time": info["train_time"],
                "remote_memory": 0 if remote_memory is None else remote_memory.length(),
            }
            if "sync" in info:
                d["sync"] = info["sync"]

            # --- info は展開して格納
            self._dict_update(d, info["train_info"], "train")
            self.log_history.append(d)

            if _time - self.log_t0 > self.train_log_interval:
                self.log_t0 = _time
                self._write_trainer_log()

        if self.enable_checkpoint:
            if _time - self.checkpoint_t0 > self.checkpoint_interval:
                self.checkpoint_t0 = _time
                self._save_parameter(info["parameter"], info["train_count"])

    def _write_trainer_log(self):
        if self.fp_dict["trainer"] is None:
            return
        if len(self.log_history) == 0:
            return

        info = self.log_history[-1]
        train_count = info["train_count"]
        remote_memory = info["remote_memory"]

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "remote_memory": remote_memory,
            "train_count": train_count,
        }
        if "sync" in info:
            d["trainer_parameter_sync_count"] = info["sync"]

        for k in self.log_history[-1].keys():
            if k in [
                "remote_memory",
                "sync",
                "train_count",
            ]:
                continue

            arr = [h[k] for h in self.log_history if (k in h) and (h[k] is not None)]
            if k in d:
                k = f"info_{k}"
            d[k] = summarize_info_from_list(arr)

        self._write_log(self.fp_dict["trainer"], d)
        self.log_history = []

    def _save_parameter(
        self,
        parameter,
        train_count,
    ):
        parameter.save(os.path.join(self.param_dir, f"{train_count}.pickle"))

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, info):
        self._init_dir(info["config"])
        self._write_config_summary(info["config"])
        self._write_system_info_summary(info["config"])

        self.player_num = info["env"].player_num
        self.actor_id = info["actor_id"]

        self.fp_dict["actor"] = open(
            os.path.join(self.train_log_dir, f"actor{self.actor_id}.txt"), "w", encoding="utf-8"
        )
        if self.actor_id == 0:
            self.fp_dict["system"] = open(os.path.join(self.train_log_dir, "system.txt"), "w", encoding="utf-8")
        else:
            self.fp_dict["system"] = None
        self.fp_dict["episode"] = None

        self.log_history = []
        self.log_t0 = time.time()

    def on_episodes_end(self, info):
        if self.enable_train_log:
            self._write_actor_log()
        self.close()

    def on_episode_begin(self, info):
        self.history_step = []

        if self.actor_id == 0 and self.enable_episode_log:
            if self.fp_dict["episode"] is not None:
                self.fp_dict["episode"].close()
                self.fp_dict["episode"] = None
            episode_count = info["episode_count"]
            self.fp_dict["episode"] = open(
                os.path.join(self.episode_log_dir, f"episode{episode_count}.txt"), "w", encoding="utf-8"
            )
            self.episode_info_env = {}
            self.episode_info_worker = {}

    def on_step_action_before(self, info) -> None:
        if self.actor_id == 0 and self.enable_episode_log:
            self._tmp_episode_env(info)

    def on_step_begin(self, info) -> None:
        if self.actor_id == 0 and self.enable_episode_log:
            self._tmp_episode_worker(info)
            self._write_episode_log()

    def on_skip_step(self, info) -> None:
        if self.actor_id == 0 and self.enable_episode_log:
            self._tmp_episode_env(info)
            self._write_episode_log()

    def on_step_end(self, info):
        if self.enable_train_log:
            # --- info は展開して格納
            d = {
                "step_time": info["step_time"],
                "train_time": info["train_time"],
            }
            self._dict_update(d, info["env"].info, "env")
            self._dict_update(d, info["train_info"], "train")
            for i, w in enumerate(info["workers"]):
                self._dict_update(d, w.info, f"work{i}")
            self.history_step.append(d)

        _time = time.time()
        if _time - self.log_t0 > self.train_log_interval:
            self.log_t0 = _time
            self._write_actor_log()
            self._write_system_log(info["config"])

    def on_episode_end(self, info):
        if self.actor_id == 0 and self.enable_episode_log:
            self._tmp_episode_env(info)
            self._write_episode_log()
            if self.fp_dict["episode"] is not None:
                self.fp_dict["episode"].close()
                self.fp_dict["episode"] = None

        if len(self.history_step) == 0:
            return
        if not self.enable_train_log:
            return

        remote_memory = info["remote_memory"]
        trainer = info["trainer"]
        episode_rewards = info["episode_rewards"]
        worker_indices = info["worker_indices"]

        d = {
            "episode_count": info["episode_count"],
            "episode_step": info["episode_step"],
            "episode_time": info["episode_time"],
            "remote_memory": 0 if remote_memory is None else remote_memory.length(),
            "train_count": 0 if trainer is None else trainer.get_train_count(),
        }
        if info.get("eval_rewards", None) is not None:
            for i, r in enumerate(info["eval_rewards"]):
                d[f"eval_reward{i}"] = r
            if "sync" in info:
                d["worker_parameter_sync_count"] = info["sync"]

        rewards = [episode_rewards[worker_indices[i]] for i in range(self.player_num)]
        for i, r in enumerate(rewards):
            d[f"episode_reward{i}"] = r
        for k in self.history_step[-1].keys():
            arr = [h[k] for h in self.history_step if (k in h) and (h[k] is not None)]
            if k in d:
                k = f"info_{k}"
            d[k] = summarize_info_from_list(arr)

        self.log_history.append(d)

    def _write_actor_log(self):
        if self.fp_dict["actor"] is None:
            return
        if len(self.log_history) == 0:
            return

        d = {
            "date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "episode_count": self.log_history[-1]["episode_count"],
            "remote_memory": self.log_history[-1]["remote_memory"],
            "train_count": self.log_history[-1]["train_count"],
        }
        for k in self.log_history[-1].keys():
            if k in [
                "episode_count",
                "remote_memory",
                "train_count",
            ]:
                continue
            arr = [h[k] for h in self.log_history if (k in h) and (h[k] is not None)]
            if k in d:
                k = f"info_{k}"
            d[k] = summarize_info_from_list(arr)

        self._write_log(self.fp_dict["actor"], d)
        self.log_history = []

    def _write_system_log(self, config: Config):
        if self.fp_dict["system"] is None:
            return

        d: Dict[str, Any] = {"date": dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}
        if config.enable_ps:
            try:
                import psutil

                d["memory"] = psutil.virtual_memory().percent
                cpus = cast(List[float], psutil.cpu_percent(percpu=True))
                for i, cpu in enumerate(cpus):
                    d[f"cpu_{i}"] = cpu
                d["cpu"] = np.mean(cpus)
                d["cpu_num"] = len(cpus)
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
                        gpu += rate.gpu
                        gpu_memory += rate.memory
                    d["gpu"] = gpu / gpu_num
                    d["gpu_memory"] = gpu_memory / gpu_num
                    d["gpu_num"] = gpu_num
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
        if self.add_render:
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
            if self.add_render:
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

    # ---------------------------
    # Game
    # ---------------------------
    def on_game_init(self, info) -> None:
        env_config = info["env_config"]
        config = Config(env_config, rl_config=None)
        self._init_dir(config)
        self.fp_dict["episode"] = None
        self.episode_count = 0
        self.episode_info_env = {}
        self.episode_info_worker = {}

    def on_game_begin(self, info) -> None:
        if self.fp_dict["episode"] is not None:
            self.fp_dict["episode"].close()
            self.fp_dict["episode"] = None
        self.fp_dict["episode"] = open(
            os.path.join(self.episode_log_dir, f"episode{self.episode_count}.txt"), "w", encoding="utf-8"
        )
        self.episode_info_env = {}
        self.episode_info_worker = {}
        self._tmp_episode_env(info)
        self._write_episode_log()

    def on_game_end(self, info) -> None:
        self.episode_count += 1
        if self.fp_dict["episode"] is not None:
            self.fp_dict["episode"].close()
            self.fp_dict["episode"] = None

    def on_game_step_end(self, info) -> None:
        self._tmp_episode_env(info)
        self._tmp_episode_worker(info)
        self._write_episode_log()
