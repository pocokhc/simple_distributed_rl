import datetime as dt
import logging
import random
import time
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner import sequence
from srl.runner.sequence import Config
from srl.utils.common import listdictdict_to_dictlist, to_str_time

logger = logging.getLogger(__name__)


class TrainerCallback(ABC):
    def on_trainer_start(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_train(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_end(self, **kwargs) -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, **kwargs) -> bool:
        return False


@dataclass
class TrainerPrintProgress(TrainerCallback):
    max_progress_time: int = 60 * 10  # s
    start_progress_timeout: int = 5  # s

    def __post_init__(self):
        assert self.start_progress_timeout > 0
        assert self.start_progress_timeout < self.max_progress_time
        self.progress_timeout = self.start_progress_timeout

    def on_trainer_start(
        self,
        max_train_count,
        timeout,
        remote_memory,
        **kwargs,
    ) -> None:
        self.max_train_count = max_train_count
        self.timeout = timeout

        self.progress_t0 = self.t0 = time.time()
        self.progress_history = []

        self.train_time = 0
        self.train_count = 0

        print(
            "### max train: {}, timeout: {}, memory len: {}".format(
                max_train_count,
                to_str_time(timeout),
                remote_memory.length(),
            )
        )

    def on_trainer_end(self, **kwargs) -> None:
        self._trainer_print_progress()

    def on_trainer_train(
        self,
        trainer,
        train_info,
        train_count,
        train_time,
        valid_reward,
        **kwargs,
    ) -> None:
        self.train_time = train_time
        self.train_count = train_count

        self.progress_history.append(
            {
                "train_count": train_count,
                "train_time": train_time,
                "train_info": train_info,
                "trainer_train_count": trainer.get_train_count(),
                "valid_reward": valid_reward,
            }
        )
        if self._check_print_progress():
            self._trainer_print_progress()

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = _time

        # 表示間隔を増やす
        self.progress_timeout *= 2
        if self.progress_timeout > self.max_progress_time:
            self.progress_timeout = self.max_progress_time

        return True

    def _trainer_print_progress(self):
        elapsed_time = time.time() - self.t0

        s = dt.datetime.now().strftime("%H:%M:%S")
        s += f" {to_str_time(elapsed_time)}"
        s += " {:6d}tr".format(self.train_count)

        if len(self.progress_history) == 0:
            pass  # 多分来ない
        else:
            train_time = np.mean([h["train_time"] for h in self.progress_history])

            # 残り時間
            if self.max_train_count > 0:
                remain_train = (self.max_train_count - self.train_count) * train_time
            else:
                remain_train = np.inf
            if self.timeout > 0:
                remain_time = self.timeout - elapsed_time
            else:
                remain_time = np.inf
            remain = min(remain_train, remain_time)
            s += f" {to_str_time(remain)}(remain)"

            # train time
            s += f", {train_time:.4f}s/tr"

            # valid
            valid_rewards = [h["valid_reward"] for h in self.progress_history if h["valid_reward"] is not None]
            if len(valid_rewards) > 0:
                s += f", {np.mean(valid_rewards):.3f} val_rew"

            # train info
            d = listdictdict_to_dictlist(self.progress_history, "train_info")
            for k, arr in d.items():
                s += f"|{k} {np.mean(arr):.4f}"

        print(s)
        self.progress_history = []


# ---------------------------------
# train only
# ---------------------------------
def train_only(
    config: Config,
    parameter: Optional[RLParameter],
    remote_memory: RLRemoteMemory,
    # train config
    max_train_count: int = -1,
    timeout: int = -1,
    enable_validation: bool = True,
    seed: Optional[int] = None,
    # print
    print_progress: bool = True,
    max_progress_time: int = 60 * 10,  # s
    print_progress_kwargs: Optional[dict] = None,
    # log TODO
    # enable_file_logger: bool = True,
    # file_logger_kwargs: Optional[dict] = None,
    # remove_file_logger: bool = True,
    # other
    # callbacks: List[Callback] = None,
) -> Tuple[RLParameter, RLRemoteMemory, object]:

    assert max_train_count > 0 or timeout > 0
    assert remote_memory.length() > 0

    config = config.copy(env_share=False)
    config.training = True
    config.enable_validation = enable_validation

    if config.seed is None:
        config.seed = seed

    callbacks = []
    if print_progress:
        if print_progress_kwargs is None:
            print_progress_kwargs = {}
        callbacks.append(TrainerPrintProgress(max_progress_time=max_progress_time, **print_progress_kwargs))

    # -----------------------------
    config.assert_params()

    if parameter is None:
        parameter = config.make_parameter()

    # valid
    if config.enable_validation:
        valid_config = config.copy(env_share=False)
        valid_config.enable_validation = False
        valid_config.players = config.validation_players
        valid_config.rl_config.remote_memory_path = ""
        env = valid_config.make_env()
    else:
        env = None

    # random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

        import tensorflow as tf

        tf.random.set_seed(config.seed)
        if env is not None:
            env.set_seed(config.seed)

    # --- trainer
    trainer = config.make_trainer(parameter, remote_memory)

    # callback
    _params = {
        "config": config,
        "max_train_count": max_train_count,
        "timeout": timeout,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "train_count": 0,
    }
    [c.on_trainer_start(**_params) for c in callbacks]

    t0 = time.time()
    train_count = 0
    while True:
        train_t0 = time.time()
        train_info = trainer.train()
        train_time = time.time() - train_t0
        train_count += 1

        # validation
        valid_reward = None
        if config.enable_validation:
            if train_count % (config.validation_interval + 1) == 0:
                rewards = sequence.evaluate(
                    valid_config,
                    parameter=parameter,
                    max_episodes=config.validation_episode,
                )
                if env.player_num > 1:
                    rewards = [r[config.validation_player] for r in rewards]
                valid_reward = np.mean(rewards)

        # callbacks
        _params["train_info"] = train_info
        _params["train_count"] = train_count
        _params["train_time"] = train_time
        _params["valid_reward"] = valid_reward
        [c.on_trainer_train(**_params) for c in callbacks]

        if max_train_count > 0 and train_count > max_train_count:
            break
        if timeout > 0 and time.time() - t0 > timeout:
            break

        # callback end
        if True in [c.intermediate_stop(**_params) for c in callbacks]:
            break

    # callbacks
    [c.on_trainer_end(**_params) for c in callbacks]

    # -----------------------------

    history = None  # TODO
    return parameter, remote_memory, history
