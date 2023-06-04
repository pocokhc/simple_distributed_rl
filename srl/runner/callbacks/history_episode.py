import glob
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import srl
from srl.base.env.env_run import EnvRun
from srl.base.rl.worker_run import WorkerRun
from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.utils.common import JsonNumpyEncoder

logger = logging.getLogger(__name__)


@dataclass
class HistoryEpisode(Callback):
    save_dir: str

    def __post_init__(self):
        self.fp: Optional[io.TextIOWrapper] = None

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"create dirs: {self.save_dir}")

        # --- create dir
        self.episodes_dir = os.path.join(self.save_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)
        logger.info(f"create episodes_dir: {self.episodes_dir}")

    def __del__(self):
        self.close()

    def close(self):
        if self.fp is not None:
            try:
                self.fp.close()
            finally:
                self.fp = None

    def _write_log(self, fp, d):
        fp.write(json.dumps(d, cls=JsonNumpyEncoder) + "\n")
        # fp.flush()

    def _init(self, config: Config):
        # --- ver
        with open(os.path.join(self.save_dir, "version.txt"), "w", encoding="utf-8") as f:
            f.write(srl.__version__)

        # --- config
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    # ---------------------------
    # episode
    # ---------------------------
    def on_episodes_begin(self, info):
        self._init(info["config"])
        self.actor_id = info["config"].actor_id

    def on_episodes_end(self, info):
        self.close()

    def on_episode_begin(self, info):
        self.close()
        episode_count = info["episode_count"]
        self.fp = open(
            os.path.join(self.episodes_dir, f"episode{episode_count}.txt"),
            "w",
            encoding="utf-8",
        )
        self.step_info_env = {}
        self.step_info_worker = {}

    def on_step_action_before(self, info) -> None:
        if self.actor_id == 0:
            self._tmp_step_env(info)

    def on_step_begin(self, info) -> None:
        if self.actor_id == 0:
            self._tmp_step_worker(info)
            self._write_step()

    def on_skip_step(self, info) -> None:
        if self.actor_id == 0:
            self._tmp_step_env(info)
            self._write_step(is_skip_step=True)

    def on_episode_end(self, info):
        if self.actor_id == 0:
            self._tmp_step_env(info)
            self._write_step(is_skip_step=True)
        self.close()

    def _tmp_step_env(self, info):
        if self.fp is None:
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
        # --- render info
        d["env_rgb_array"] = env.render_rgb_array()

        self.step_info_env = d

    def _tmp_step_worker(self, info):
        if self.fp is None:
            return
        d = {
            "action": info["action"],
        }
        env: EnvRun = info["env"]
        workers: List[WorkerRun] = info["workers"]
        for i, w in enumerate(workers):
            d[f"work{i}_info"] = w.info
            d[f"work{i}_rgb_array"] = w.render_rgb_array(env)

        self.step_info_worker = d

    def _write_step(self, is_skip_step=False):
        if self.fp is None:
            return
        d = {"is_skip_step": is_skip_step}
        d.update(self.step_info_env)
        d.update(self.step_info_worker)
        self._write_log(self.fp, d)

    # ---------------------------
    # read
    # ---------------------------
    def read_config(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d

    def read_episode(self, path: str) -> List[dict]:
        if not os.path.isfile(path):
            return []

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    data.append(d)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError {e.args[0]}, '{line.strip()}', '{path}'")
        return data

    # ---------------------------
    # replay
    # ---------------------------
    def replay(self, _is_test: bool = False):
        self.episode_files = glob.glob(os.path.join(self.episodes_dir, "episode*.txt"))
        self.episode_cache = {}
        self.config = self.read_config(os.path.join(self.save_dir, "config.json"))
        self.player_num = self.config["env_config"]["player_num"]

        from srl.runner.game_window import EpisodeReplay

        EpisodeReplay(self, _is_test=_is_test).play()

    @property
    def episode_num(self) -> int:
        return len(self.episode_files)

    def load_episode(self, episode: int) -> Tuple[dict, List[dict]]:
        if not (0 <= episode < len(self.episode_files)):
            return {}, []

        if episode in self.episode_cache:
            return self.episode_cache[episode]

        logger.info(f"Start loading episode{episode}.")
        t0 = time.time()

        steps = self.read_episode(self.episode_files[episode])

        total_reward = np.zeros(self.player_num)
        for step in steps:
            if "rewards" in step:
                step["rewards"] = np.array(step["rewards"])
                total_reward += step["rewards"]
            if "env_rgb_array" in step:
                step["env_rgb_array"] = np.array(step["env_rgb_array"])
            for i in range(self.player_num):
                if f"work{i}_rgb_array" in step:
                    step[f"work{i}_rgb_array"] = np.array(step[f"work{i}_rgb_array"])
        episode_info = {
            "total_rewards": total_reward,
        }
        self.episode_cache[episode] = (episode_info, steps)

        logger.info(f"Episode loaded.({time.time()-t0:.1f}s)")
        return episode_info, steps
