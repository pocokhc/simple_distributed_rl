import datetime
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback, CallbackType, MPCallback, TrainerCallback
from srl.runner.runner import Runner
from srl.utils.util_str import to_str_info, to_str_reward, to_str_time

logger = logging.getLogger(__name__)


# 進捗に対して表示、少しずつ間隔を長くする(上限あり)
@dataclass
class PrintProgress(Callback, MPCallback, TrainerCallback):
    start_time: int = 1
    interval_limit: int = 60 * 10
    progress_env_info: bool = False
    progress_train_info: bool = True
    progress_worker_info: bool = True
    progress_worker: int = 0
    progress_max_actor: int = 5
    enable_eval: bool = True
    eval_env_sharing: bool = False
    eval_episode: int = 10
    eval_timeout: int = -1
    eval_max_steps: int = -1
    eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    eval_shuffle_player: bool = False
    eval_seed: Optional[int] = None
    eval_used_device_tf: str = "/CPU"
    eval_used_device_torch: str = "cpu"
    eval_callbacks: List[CallbackType] = field(default_factory=list)

    def __post_init__(self):
        assert self.start_time > 0
        assert self.interval_limit > self.start_time
        if self.start_time > self.interval_limit:
            logger.info(f"change start_time: {self.start_time}s -> {self.interval_limit}s")
            self.start_time = self.interval_limit

        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

    def _check_print_progress(self):
        _time = time.time()
        taken_time = _time - self.progress_t0
        if taken_time < self.progress_timeout:
            return False
        self.progress_t0 = _time

        # 表示間隔を増やす、5s以下は5sに、それ以降は２倍
        if self.progress_timeout < 5:
            self.progress_timeout = 5
        else:
            self.progress_timeout *= 2
            if self.progress_timeout > self.interval_limit:
                self.progress_timeout = self.interval_limit

        return True

    def _create_eval_runner(self, runner: Runner):
        if not self.enable_eval:
            return
        self.eval_runner = runner.create_eval_runner(self.eval_env_sharing)

        # config
        self.eval_runner.config.players = self.eval_players
        self.eval_runner.config.seed = self.eval_seed
        # device
        self.eval_runner.context.used_device_tf = self.eval_used_device_tf
        self.eval_runner.context.used_device_torch = self.eval_used_device_torch
        # context
        self.eval_runner.context.max_episodes = self.eval_episode
        self.eval_runner.context.timeout = self.eval_timeout
        self.eval_runner.context.max_steps = self.eval_max_steps
        self.eval_runner.context.shuffle_player = self.eval_shuffle_player
        self.eval_runner.context.callbacks = self.eval_callbacks

    def _eval_str(self, runner: Runner) -> str:
        if not self.enable_eval:
            return ""
        self.eval_runner._play(runner.parameter, runner.remote_memory)
        eval_rewards = self.eval_runner.state.episode_rewards_list
        eval_rewards = np.mean(eval_rewards, axis=0)
        return f"({to_str_reward(eval_rewards[self.progress_worker])}eval)"

    # -----------------------------------------------------

    def on_episodes_begin(self, runner: Runner):
        if runner.context.actor_id >= self.progress_max_actor:
            return
        context = runner.context
        self.progress_timeout = self.start_time

        # 分散の場合はactor_id=0のみevalをする
        if runner.context.distributed:
            self.enable_eval = self.enable_eval and (context.actor_id == 0)

        self._create_eval_runner(runner)

        if not context.distributed:
            print(
                "### env: {}, rl: {}, max episodes: {}, timeout: {}, max steps: {}, max train: {}".format(
                    runner.config.env_config.name,
                    runner.config.rl_config.getName(),
                    context.max_episodes,
                    to_str_time(context.timeout),
                    context.max_steps,
                    context.max_train_count,
                )
            )

        _time = time.time()
        self.progress_t0 = _time
        self.progress_history = []

        self.t0_print_time = _time
        self.t0_step_count = 0
        self.t0_episode_count = 0

    def on_episodes_end(self, runner: Runner):
        if runner.context.actor_id >= self.progress_max_actor:
            return
        self._print_actor(runner)

    def on_step_end(self, runner: Runner):
        if runner.context.actor_id >= self.progress_max_actor:
            return
        if self._check_print_progress():
            self._print_actor(runner)

    def on_episode_end(self, runner: Runner):
        if runner.context.actor_id >= self.progress_max_actor:
            return
        env = runner.state.env
        assert env is not None

        # print_workerの報酬を記録する
        player_idx = runner.state.worker_indices[self.progress_worker]
        episode_reward = env.episode_rewards[player_idx]

        d = {
            "episode_step": env.step_num,
            "episode_reward": episode_reward,
        }
        self.progress_history.append(d)

    # -----------------------------------------

    def _print_actor(self, runner: Runner):
        context = runner.context
        state = runner.state
        assert state.env is not None

        _time = time.time()
        elapsed_time = _time - state.elapsed_t0

        # [TIME] [actor] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        if context.distributed:
            s += f" actor{context.actor_id:2d}:"
        s += f" {to_str_time(elapsed_time)}"

        # calc time
        diff_step = state.total_step - self.t0_step_count
        if diff_step > 0:
            step_time = (_time - self.t0_print_time) / diff_step
        else:
            step_time = np.inf
        diff_episode = state.episode_count - self.t0_episode_count
        if diff_episode:
            episode_time = (_time - self.t0_print_time) / diff_episode
        else:
            episode_time = np.inf

        self.t0_print_time = _time
        self.t0_step_count = state.total_step
        self.t0_episode_count = state.episode_count

        # [remain]
        if (context.max_steps > 0) and (state.total_step > 0):
            remain_step = (context.max_steps - state.total_step) * step_time
        else:
            remain_step = np.inf
        if (context.max_episodes > 0) and (state.episode_count > 0):
            remain_episode = (context.max_episodes - state.episode_count) * episode_time
        else:
            remain_episode = np.inf
        if context.timeout > 0:
            remain_time = context.timeout - elapsed_time
        else:
            remain_time = np.inf
        remain_train = np.inf
        if state.trainer is not None:
            train_count = state.trainer.get_train_count()
            if (context.max_train_count > 0) and (train_count > 0):
                remain_train = (context.max_train_count - train_count) * step_time
        remain = min(min(min(remain_step, remain_episode), remain_time), remain_train)
        if remain == np.inf:
            s += "(     - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [all step] [all episode] [train]
        s += f" {state.total_step:7d}st({state.episode_count:6d}ep)"
        if state.trainer is not None:
            s += " {:5d}tr".format(state.trainer.get_train_count())

        # [sync]
        if context.distributed:
            s += f", {state.sync_actor:2d}recv"

        if diff_episode == 0:
            if diff_step == 0:
                # ---------------------------
                # no info
                # ---------------------------
                s += "1 step is not over."
            else:
                # ---------------------------
                # steps info
                # ---------------------------
                # [episode step] [step time]
                s += f", {diff_step:5d} step"
                s += f", {step_time:.5f}s/step"

                # [reward]
                r_list = [to_str_reward(r) for r in state.env.episode_rewards]
                s += " [" + ",".join(r_list) + "]"

        else:
            # ---------------------------
            # episode info
            # ---------------------------
            # [reward]
            _r = [h["episode_reward"] for h in self.progress_history]
            _r_min = to_str_reward(min(_r))
            _r_mid = to_str_reward(float(np.mean(_r)), check_skip=True)
            _r_max = to_str_reward(max(_r))
            s += f",{_r_min} {_r_mid} {_r_max} re"

            # [eval reward]
            if context.actor_id == 0:
                s += self._eval_str(runner)
            elif context.distributed:
                s += " " * 12

            # [mean episode step] [episode time]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {int(np.mean(_s)):3d}step"
            s += f", {episode_time:.2f}s/ep"

        # [memory]
        if state.remote_memory is not None:
            s += f", {state.remote_memory.length()}mem"

        # [system]
        s += self._stats_str(runner)

        # [info] , 速度優先して一番最新の状態をそのまま表示
        env_types = state.env.info_types
        rl_types = runner.config.rl_config.info_types
        if self.progress_env_info:
            s += to_str_info(state.env.info, env_types)
        if self.progress_worker_info:
            s += to_str_info(state.workers[self.progress_worker].info, rl_types)
        if self.progress_train_info:
            s += to_str_info(state.train_info, rl_types)

        print(s)
        self.progress_history = []

    def _stats_str(self, runner: Runner) -> str:
        if not runner.config.enable_stats:
            return ""

        # ,CPU100% M100%,GPU0 100% M100%
        s = ""
        if runner.context.actor_id == 0:
            if runner.context.used_psutil:
                try:
                    memory_percent, cpu_percent = runner.read_psutil()
                    s += f"[CPU{cpu_percent:3.0f}%,M{memory_percent:2.0f}%]"
                except Exception:
                    logger.debug(traceback.format_exc())
                    s += "[CPU Nan%]"

            if runner.context.used_nvidia:
                try:
                    gpus = runner.read_nvml()
                    # device_id, rate.gpu, rate.memory
                    s += "".join([f"[GPU{g[0]} {g[1]:2.0f}%,M{g[2]:2.0f}%]" for g in gpus])
                except Exception:
                    logger.debug(traceback.format_exc())
                    s += ",GPU Nan%"
        else:
            if runner.context.used_psutil:
                try:
                    memory_percent, cpu_percent = runner.read_psutil()
                    s += f"[CPU{cpu_percent:3.0f}%]"
                except Exception:
                    logger.debug(traceback.format_exc())
                    s += "[CPU Nan%]"
        return s

    # ----------------------------------
    # trainer
    # ----------------------------------
    def on_trainer_start(self, runner: Runner) -> None:
        # 分散の場合はevalをしない
        if runner.context.distributed:
            self.enable_eval = False

        self.progress_timeout = self.start_time

        self._create_eval_runner(runner)

        if not runner.context.distributed:
            assert runner.state.remote_memory is not None
            print(
                "### max train: {}, timeout: {}, memory len: {}".format(
                    runner.context.max_train_count,
                    to_str_time(runner.context.timeout),
                    runner.state.remote_memory.length(),
                )
            )

        # --- init
        _time = time.time()
        self.progress_t0 = _time
        self.progress_history = []

        self.t0_train_time = _time
        self.t0_train_count = 0

    def on_trainer_end(self, runner: Runner) -> None:
        self._print_trainer(runner)

    def on_trainer_train(self, runner: Runner) -> None:
        if self._check_print_progress():
            self._print_trainer(runner)

    def _print_trainer(self, runner: Runner):
        context = runner.context
        state = runner.state
        assert state.trainer is not None

        _time = time.time()
        elapsed_time = _time - state.elapsed_t0

        # --- head
        # [TIME] [trainer] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        if context.distributed:
            s += " trainer:"
        s += f" {to_str_time(elapsed_time)}"

        # calc time
        train_count = state.trainer.get_train_count()
        diff_count = train_count - self.t0_train_count
        if diff_count > 0:
            train_time = (_time - self.t0_train_time) / diff_count
        else:
            train_time = np.inf

        self.t0_train_time = _time
        self.t0_train_count = train_count

        # [remain]
        if (context.max_train_count > 0) and (train_count > 0):
            remain_train = (context.max_train_count - train_count) * train_time
        else:
            remain_train = np.inf
        if context.timeout > 0:
            remain_time = context.timeout - elapsed_time
        else:
            remain_time = np.inf
        remain = min(remain_train, remain_time)
        if remain == np.inf:
            s += "(     - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [train count]
        s += " {:6d}tr".format(train_count)

        if train_count == 0:
            # --- no info
            s += " 1train is not over."
        else:
            # --- train info
            # [train time]
            s += f", {train_time:.3f}s/tr"

            # [sync]
            if context.distributed:
                s += f", {state.sync_trainer:2d}send"

            # [eval]
            s += self._eval_str(runner)

            # [system]
            s += self._stats_str(runner)

            # [info] , 速度優先して一番最新の状態をそのまま表示
            if self.progress_train_info:
                s += to_str_info(state.train_info, runner.rl_config.info_types)

        print(s)
        self.progress_history = []

    # ----------------------------------
    # mp
    # ----------------------------------
    def on_start(self, runner: Runner):
        print(
            "### env: {}, rl: {}, max train: {}, timeout: {}".format(
                runner.env_config.name,
                runner.rl_config.getName(),
                runner.context.max_train_count,
                to_str_time(runner.context.timeout),
            )
        )
