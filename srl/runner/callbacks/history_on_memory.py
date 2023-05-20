import logging
import time
from dataclasses import dataclass

from srl.runner.callback import Callback, GameCallback
from srl.utils.common import summarize_info_from_dictlist

logger = logging.getLogger(__name__)


@dataclass
class HistoryOnMemory(Callback, GameCallback):
    """memory上に保存する、distributeでは実行しない"""

    def __post_init__(self):
        pass

    def on_episodes_begin(self, info):
        self.logs = []
        self.t0 = time.time()

    def on_episode_begin(self, info):
        self.infos = {}
        self._add_info("env", info["env"].info)

    def on_step_end(self, info):
        self._add_info("env", info["env"].info)
        self._add_info("trainer", info["train_info"])
        [self._add_info(f"actor0_worker{i}", w.info) for i, w in enumerate(info["workers"])]

    def _add_info(self, prefix, dict_):
        if dict_ is None:
            return
        for k, v in dict_.items():
            k = f"{prefix}_{k}"
            if k not in self.infos:
                self.infos[k] = [v]
            else:
                self.infos[k].append(v)

    def on_episode_end(self, info):
        d = summarize_info_from_dictlist(self.infos)
        d["time"] = time.time() - self.t0
        d["index"] = info["episode_count"]
        d["actor0_episode"] = info["episode_count"]
        d["actor0_episode_time"] = info["episode_time"]
        d["actor0_episode_step"] = info["episode_step"]
        for i, r in enumerate(info["episode_rewards"]):
            d[f"actor0_episode_reward{i}"] = r
        if info.get("eval_rewards", None) is not None:
            for i, r in enumerate(info["eval_rewards"]):
                d[f"actor0_eval_reward{i}"] = r

        trainer = info["trainer"]
        if trainer is not None:
            remote_memory = info["remote_memory"]
            d["train"] = trainer.get_train_count()
            d["remote_memory"] = 0 if remote_memory is None else remote_memory.length()

        self.logs.append(d)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, info):
        self.logs = []
        self.t0 = time.time()

    def on_trainer_train(self, info):
        # TODO
        d = {}
        for k, v in info["train_info"].items():
            d[f"trainer_{k}"] = v
        if info.get("eval_rewards", None) is not None:
            for i, r in enumerate(info["eval_rewards"]):
                d[f"eval_reward{i}"] = r
        d["time"] = time.time() - self.t0
        d["index"] = round(d["time"])
        self.logs.append(d)
