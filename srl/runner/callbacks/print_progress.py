import datetime
import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np

from srl.base.context import RunContext
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callback import RunnerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner
from srl.utils.util_str import to_str_info, to_str_reward, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class PrintProgress(RunnerCallback, RunCallback, TrainerCallback, Evaluate):
    """時間に対して少しずつ長く表示、学習し始めたら長くしていく"""

    # mode: str = "simple"
    start_time: int = 1
    interval_limit: int = 60 * 10
    single_line: bool = True
    progress_env_info: bool = False
    progress_train_info: bool = True
    progress_worker_info: bool = True
    progress_worker: int = 0
    progress_max_actor: int = 5

    def __post_init__(self):
        assert self.start_time > 0
        assert self.interval_limit >= self.start_time
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

    def _update_progress(self):
        # 表示間隔を増やす、5s以下は5sに、それ以降は２倍
        if self.progress_timeout < 5:
            self.progress_timeout = 5
        else:
            self.progress_timeout *= 2
        if self.progress_timeout > self.interval_limit:
            self.progress_timeout = self.interval_limit

    def _eval_str(self, context: RunContext, parameter: RLParameter) -> str:
        assert self.runner is not None
        if not self.setup_eval_runner(self.runner):
            return ""
        if context.distributed:
            if context.actor_id == 0:
                eval_rewards = self.run_eval(parameter)
                if eval_rewards is None:
                    return " " * 12
                else:
                    return f"({to_str_reward(eval_rewards[self.progress_worker])}eval)"
            else:
                return " " * 12
        else:
            eval_rewards = self.run_eval(parameter)
            if eval_rewards is None:
                return " " * 12
            else:
                return f"({to_str_reward(eval_rewards[self.progress_worker])}eval)"

    def on_runner_start(self, runner: Runner) -> None:
        s = f"### env: {runner.env_config.name}, rl: {runner.rl_config.get_name()}"
        if runner.context.max_episodes > 0:
            s += f", max episodes: {runner.context.max_episodes}"
        if runner.context.timeout > 0:
            s += f", timeout: {to_str_time(runner.context.timeout)}"
        if runner.context.max_steps > 0:
            s += f", max steps: {runner.context.max_steps}"
        if runner.context.max_train_count > 0:
            s += f", max train: {runner.context.max_train_count}"
        if runner.context.max_memory > 0:
            s += f", max memory: {runner.context.max_memory}"
        print(s)

    # -----------------------------------------------------
    # actor
    # -----------------------------------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        if context.actor_id >= self.progress_max_actor:
            return

        # 分散の場合はactor_id=0のみevalをする
        if context.distributed:
            self.enable_eval = self.enable_eval and (context.actor_id == 0)

        self.progress_timeout = self.start_time

        _time = time.time()
        self.progress_t0 = _time
        self.progress_history = []

        self.t0_print_time = _time
        self.t0_step_count = 0
        self.t0_episode_count = 0
        self.t0_memory_count = 0
        self.t0_actor_send_q = 0

    def on_episodes_end(self, context: RunContext, state: RunStateActor):
        if context.actor_id >= self.progress_max_actor:
            return
        self._print_actor(context, state)

    def on_step_end(self, context: RunContext, state: RunStateActor):
        if context.actor_id >= self.progress_max_actor:
            return
        if time.time() - self.progress_t0 > self.progress_timeout:
            self._print_actor(context, state)
            self._update_progress()
            self.progress_t0 = time.time()  # last

    def on_episode_end(self, context: RunContext, state: RunStateActor):
        if context.actor_id >= self.progress_max_actor:
            return

        # print_workerの報酬を記録する
        player_idx = state.worker_indices[self.progress_worker]
        episode_reward = state.env.episode_rewards[player_idx]

        d = {
            "episode_step": state.env.step_num,
            "episode_reward": episode_reward,
        }
        self.progress_history.append(d)

    # -----------------------------------------

    def _print_actor(self, context: RunContext, state: RunStateActor):
        _time = time.time()
        elapsed_time = _time - state.elapsed_t0

        # [TIME] [actor] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        if context.distributed:
            s += f" actor{context.actor_id:2d}:"
        s += f" {to_str_time(elapsed_time)}"

        # calc time
        diff_time = _time - self.t0_print_time
        if diff_time < 0.1:
            diff_time = 0.1
        diff_step = state.total_step - self.t0_step_count
        diff_episode = state.episode_count - self.t0_episode_count
        step_time = diff_time / diff_step if diff_step > 0 else np.inf
        episode_time = diff_time / diff_episode if diff_episode > 0 else np.inf
        self.t0_print_time = _time
        self.t0_step_count = state.total_step
        self.t0_episode_count = state.episode_count

        # calc memory
        memory_time = np.inf
        if state.memory is not None:
            diff_memory = state.memory.length() - self.t0_memory_count
            if diff_memory > 0:
                memory_time = diff_time / diff_memory
            self.t0_memory_count = state.memory.length()

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
        remain_memory = np.inf
        if state.memory is not None:
            if context.max_memory > 0 and state.memory.length() > 0:
                remain_memory = (context.max_memory - state.memory.length()) * memory_time
        remain = min(min(min(remain_step, remain_episode), remain_time), remain_train)
        remain = min(remain, remain_memory)
        if remain == np.inf:
            s += "(     - left)"
        else:
            s += f"({to_str_time(remain)} left)"

        # [step time]
        _c = diff_step / diff_time
        if _c < 10:
            s += f",{_c:6.2f}st/s"
        else:
            s += f",{int(_c):6d}st/s"

        # [all step]
        s += f",{state.total_step:7d}st"

        # [memory]
        if state.memory is not None:
            s += f",{state.memory.length():6d}mem"

        # [sync]
        if context.distributed:
            diff_q = state.actor_send_q - self.t0_actor_send_q
            s += f", Q {int(diff_q / diff_time):4d}send/s({state.actor_send_q:8d})"
            self.t0_actor_send_q = state.actor_send_q
            s += f", {state.sync_actor:4d} recv Param"

        # [all episode] [train]
        s += f", {state.episode_count:3d}ep"
        if state.trainer is not None:
            s += ", {:5d}tr".format(state.trainer.get_train_count())

        if diff_episode <= 0:
            if diff_step <= 0:
                # --- no info
                s += "1 step is not over."
            else:
                # --- steps info
                # [episode step]
                s += f", {state.env.step_num}st"

                # [reward]
                r_list = [to_str_reward(r) for r in state.env.episode_rewards]
                s += " [" + ",".join(r_list) + "]reward"

        else:
            # --- episode info
            # [mean episode step]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f", {int(np.mean(_s)):3d}st"

            # [reward]
            _r = [h["episode_reward"] for h in self.progress_history]
            _r_min = to_str_reward(min(_r))
            _r_mid = to_str_reward(float(np.mean(_r)), check_skip=True)
            _r_max = to_str_reward(max(_r))
            s += f",{_r_min} {_r_mid} {_r_max} re"

            # [eval reward]
            s += self._eval_str(context, state.parameter)

        # [system]
        s += self._stats_str()

        # [info] , 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        stypes = state.env.info_types
        rl_types = state.worker.config.get_info_types()
        if self.progress_env_info:
            s_info += to_str_info(state.env.info, stypes)
        if self.progress_worker_info:
            s_info += to_str_info(state.workers[self.progress_worker].info, rl_types)
        if self.progress_train_info:
            if state.trainer is not None:
                s_info += to_str_info(state.trainer.get_info(), rl_types)

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)
        self.progress_history = []

    def _stats_str(self) -> str:
        if self.runner is None:
            return ""
        if not self.runner.config.enable_stats:
            return ""

        # ,CPU100% M100%,GPU0 100% M100%
        s = ""
        if self.runner.context.actor_id == 0:
            try:
                memory_percent, cpu_percent = self.runner.read_psutil()
                if memory_percent != np.NaN:
                    s += f"[CPU{cpu_percent:3.0f}%,M{memory_percent:2.0f}%]"
            except Exception:
                logger.debug(traceback.format_exc())
                s += "[CPU Nan%]"

            try:
                gpus = self.runner.read_nvml()
                # device_id, rate.gpu, rate.memory
                s += "".join([f"[GPU{g[0]} {g[1]:2.0f}%,M{g[2]:2.0f}%]" for g in gpus])
            except Exception:
                logger.debug(traceback.format_exc())
                s += ",GPU Nan%"
        else:
            try:
                memory_percent, cpu_percent = self.runner.read_psutil()
                if memory_percent != np.NaN:
                    s += f"[CPU{cpu_percent:3.0f}%]"
            except Exception:
                logger.debug(traceback.format_exc())
                s += "[CPU Nan%]"
        return s

    # ----------------------------------
    # trainer
    # ----------------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer) -> None:
        # eval, 分散の場合はevalをしない
        if context.distributed:
            self.enable_eval = False

        self.progress_timeout = self.start_time

        _time = time.time()
        self.progress_t0 = _time
        self.progress_history = []

        self.t0_train_time = _time
        self.t0_train_count = 0
        self.t0_trainer_recv_q = 0

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer) -> None:
        self._print_trainer(context, state)

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer) -> None:
        if time.time() - self.progress_t0 > self.progress_timeout:
            self._print_trainer(context, state)
            self._update_progress()
            self.progress_t0 = time.time()  # last

    def _print_trainer(self, context: RunContext, state: RunStateTrainer):
        _time = time.time()
        elapsed_time = _time - state.elapsed_t0

        # --- head
        # [TIME] [trainer] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        if context.distributed:
            s += " trainer:"
        s += f" {to_str_time(elapsed_time)}"

        # calc time
        diff_time = _time - self.t0_train_time
        if diff_time < 0.1:
            diff_time = 0.1
        train_count = state.trainer.get_train_count()
        diff_train_count = train_count - self.t0_train_count
        train_time = diff_time / diff_train_count if diff_train_count > 0 else np.inf
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

        # [train time]
        _c = diff_train_count / diff_time
        if _c < 10:
            s += f",{_c:6.2f}tr/s"
        else:
            s += f",{int(_c):6d}tr/s"

        # [train count]
        s += ",{:7d}tr".format(train_count)

        # [memory]
        if state.memory is not None:
            s += f",{state.memory.length():6d}mem"

        # [distributed]
        if context.distributed:
            diff_q = state.trainer_recv_q - self.t0_trainer_recv_q
            s += f", Q {int(diff_q / diff_time):4d}recv/s({state.trainer_recv_q:8d})"
            self.t0_trainer_recv_q = state.trainer_recv_q
            s += f", {state.sync_trainer:4d} send Param"

        if train_count == 0:
            # no info
            s += " 1train is not over."
        else:
            # [eval]
            s += self._eval_str(context, state.parameter)

        # [system]
        s += self._stats_str()

        # [info] , 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        if self.progress_train_info:
            if state.trainer is not None:
                s_info += to_str_info(state.trainer.get_info(), state.trainer.config.get_info_types())

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)
        self.progress_history = []
