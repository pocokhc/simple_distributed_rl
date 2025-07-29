import datetime
import logging
import math
import time
import traceback
from dataclasses import dataclass

import numpy as np

from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.evaluate import Evaluate
from srl.utils.util_str import to_str_reward, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class PrintProgress(RunCallback, Evaluate):
    """時間に対して少しずつ長く表示、学習し始めたら長くしていく"""

    # mode: str = "simple"
    start_time: int = 1
    interval_limit: int = 60 * 2
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

        self._debug_gpu_error = False

    def _update_progress(self, reset_interval: bool = False):
        if reset_interval:
            self.progress_timeout = self.start_time
        # 表示間隔を増やす、5s以下は5sに、それ以降はn倍
        if self.progress_timeout < 5:
            self.progress_timeout = 5
        else:
            self.progress_timeout *= 1.5
        if self.progress_timeout > self.interval_limit:
            self.progress_timeout = self.interval_limit

    def _eval_str(self, context: RunContext, state) -> str:
        if not self.enable_eval:
            return ""
        if context.distributed:
            if context.actor_id == 0:
                eval_rewards = self.run_eval_with_state(context, state)
                if eval_rewards is None:
                    return " " * 12
                else:
                    return f"({to_str_reward(eval_rewards[self.progress_worker])}eval)"
            else:
                return " " * 12
        else:
            eval_rewards = self.run_eval_with_state(context, state)
            if eval_rewards is None:
                return " " * 12
            else:
                return f"({to_str_reward(eval_rewards[self.progress_worker])}eval)"

    # -----------------------------------------------------
    # actor
    # -----------------------------------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        if context.actor_id >= self.progress_max_actor:
            return

        s = "###"
        if context.distributed:
            s += f" [actor{context.actor_id:2d}]"
        if len(context.players) > 1:
            s += f", {context.players}"
        s += f" rl: {context.rl_config.get_name()}"
        if context.framework == "tensorflow":
            s += f", tf={context.used_device_tf}"
        elif context.framework == "torch":
            s += f", torch={context.used_device_torch}"
        s += f", {context.flow_mode}"
        s += f", env: {context.env_config.name}"
        if context.max_episodes > 0:
            s += f", max episodes: {context.max_episodes}"
        if context.timeout > 0:
            s += f", timeout: {to_str_time(context.timeout)}"
        if context.max_steps > 0:
            s += f", max steps: {context.max_steps}"
        if context.max_train_count > 0:
            s += f", max train: {context.max_train_count}"
        if context.max_memory > 0:
            s += f", max memory: {context.max_memory}"
        print(s)

        # 分散の場合はactor_id=0のみevalをする
        if context.distributed:
            self.enable_eval = self.enable_eval and (context.actor_id == 0)

        self.progress_timeout = self.start_time

        _time = time.time()
        self.progress_t0 = _time
        self.prev_train_count = 0
        self.progress_history = []

        self.t0_print_time = _time
        self.t0_step_count = 0
        self.t0_episode_count = 0
        self.t0_train_count = 0
        self.t0_memory_count = 0
        self.t0_actor_send_q = 0

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs):
        if context.actor_id >= self.progress_max_actor:
            return
        self._print_actor(context, state)

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs):
        if context.actor_id >= self.progress_max_actor:
            return
        if time.time() - self.progress_t0 > self.progress_timeout:
            self._print_actor(context, state)
            reset_interval = (self.prev_train_count == 0) and (state.train_count > 0)  # 0->1になる瞬間だけリセット
            self._update_progress(reset_interval)
            self.prev_train_count = state.train_count
            self.progress_t0 = time.time()  # last

    def on_episode_end(self, context: RunContext, state: RunStateActor, **kwargs):
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
        if state.trainer is not None:
            diff_train = state.train_count - self.t0_train_count
            train_time = diff_time / diff_train if diff_train > 0 else np.inf
            self.t0_train_count = state.train_count

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
            train_count = state.train_count
            if (context.max_train_count > 0) and (train_count > 0):
                remain_train = (context.max_train_count - train_count) * train_time
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
            s += f" {_c:4.2f}st/s"
        else:
            s += f" {int(_c):4d}st/s"

        # [train time]
        if state.trainer is not None:
            _c = diff_train / diff_time
            if _c < 10:
                s += f" {_c:4.2f}tr/s"
            else:
                s += f" {int(_c):4d}tr/s"

        # [memory]
        if state.memory is not None:
            s += f" {state.memory.length():5d}mem"

        # [train]
        if state.trainer is not None:
            s += " {:5d}tr".format(state.trainer.train_count)

        # [distributed]
        if context.distributed:
            diff_q = state.actor_send_q - self.t0_actor_send_q
            s += f" {math.ceil(diff_q / diff_time):4d}send/s"
            self.t0_actor_send_q = state.actor_send_q
            s += f" {state.sync_actor:4d}sync"

        # [system]
        s += self._stats_str(context)

        # [all episode]
        s += f" {state.episode_count:5d}ep"

        # [all step]
        s += f" {state.total_step:6d}st"

        if diff_episode <= 0:
            if diff_step <= 0:
                # --- no info
                s += "1 step is not over."
            else:
                # --- steps info
                # [episode step]
                s += f" {state.env.step_num:3d}st/ep"

                # [reward]
                r_list = [to_str_reward(r) for r in state.env.episode_rewards]
                if len(r_list) == 1:
                    s2 = "(ongoing) " + r_list[0]
                else:
                    s2 = "ongoing(" + ",".join(r_list) + ")"
                s += f"{s2:>21s} re"  # 6+1+6+1+6

        else:
            # --- episode info
            # [mean episode step]
            _s = [h["episode_step"] for h in self.progress_history]
            s += f" {int(np.mean(_s)):3d}st/ep"

            # [reward]
            _r = [h["episode_reward"] for h in self.progress_history]
            _r_min = to_str_reward(min(_r))
            _r_mid = to_str_reward(float(np.mean(_r)), check_skip=True)
            _r_max = to_str_reward(max(_r))
            s += f" {_r_min} {_r_mid} {_r_max} re"

            # [eval reward]
            s += self._eval_str(context, state)

        # [info] 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        if self.progress_env_info:
            s_info += state.env.info.to_str()
        if self.progress_worker_info:
            s_info += state.workers[self.progress_worker].info.to_str()
        if self.progress_train_info and state.trainer is not None:
            s_info += state.trainer.info.to_str()

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)
        self.progress_history = []

    def _stats_str(self, context: RunContext) -> str:
        if not context.enable_stats:
            return ""

        s = " "
        if context.actor_id == 0:
            try:
                from srl.base.system import psutil_

                mem = psutil_.read_memory()
                cpu = psutil_.read_cpu()
                s += f"[CPU{cpu:3.0f}%,M{mem:2.0f}%]"
            except Exception:
                logger.debug(traceback.format_exc())
                s += "[CPU error]"

            try:
                from srl.base.system.pynvml_ import read_nvml

                gpus = read_nvml()
                # device_id, rate.gpu, rate.memory
                s += "".join([f"[GPU{g[0]} {g[1]:2.0f}%,M{g[2]:2.0f}%]" for g in gpus])
            except Exception:
                if self._debug_gpu_error:
                    logger.debug(traceback.format_exc())
                else:
                    logger.info(traceback.format_exc())
                    self._debug_gpu_error = True
                s += "[GPU error]"

        else:
            try:
                from srl.base.system import psutil_

                cpu = psutil_.read_cpu()
                s += f"[CPU{cpu:3.0f}%]"
            except Exception:
                logger.debug(traceback.format_exc())
                s += "[CPU error]"
        return s

    # ----------------------------------
    # trainer
    # ----------------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        # eval, 分散の場合はevalをしない
        if context.distributed:
            self.enable_eval = False

        s = "###"
        if context.distributed:
            s += " [trainer]"
        s += f" rl: {context.rl_config.get_name()}"
        if context.framework == "tensorflow":
            s += f", tf={context.used_device_tf}"
        elif context.framework == "torch":
            s += f", torch={context.used_device_torch}"
        s += f", {context.flow_mode}"
        if context.timeout > 0:
            s += f", timeout: {to_str_time(context.timeout)}"
        if context.max_train_count > 0:
            s += f", max train: {context.max_train_count}"
        if context.max_memory > 0:
            s += f", max memory: {context.max_memory}"
        print(s)

        self.progress_timeout = self.start_time
        self.progress_t0 = time.time()

        self.t0_train_time = time.time()
        self.t0_train_count = 0
        self.t0_trainer_recv_q = 0
        self._run_print = False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        self._print_trainer(context, state)

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        if time.time() - self.progress_t0 > self.progress_timeout:
            self._print_trainer(context, state)
            self._update_progress()
            self.progress_t0 = time.time()  # last
        return False

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
        train_count = state.train_count
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
            s += f" {_c:4.2f}tr/s"
        else:
            s += f" {int(_c):4d}tr/s"

        # [memory]
        if state.memory is not None:
            s += f" {state.memory.length():5d}mem"

        # [distributed]
        if context.distributed:
            diff_q = state.trainer_recv_q - self.t0_trainer_recv_q
            s += f" {math.ceil(diff_q / diff_time):4d}recv/s"
            self.t0_trainer_recv_q = state.trainer_recv_q

            s += f" {state.sync_trainer:4d}sync"

        # [system]
        s += self._stats_str(context)

        # [all train count]
        s += " {:5d}tr".format(state.trainer.train_count)

        if train_count == 0:
            # no info
            s += " 1train is not over."
        else:
            # [eval]
            s += self._eval_str(context, state)

        # [info] , 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        if self.progress_train_info:
            s_info += state.trainer.info.to_str()

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)

    # ----------------------------------
    # memory
    # ----------------------------------
    def on_memory_start(self, context: "RunContext", info: dict, **kwargs) -> None:
        self.progress_timeout = self.start_time
        self.progress_t0 = time.time()
        self.elapsed_t0 = time.time()

        self.mem_to_train_names = [f.__name__ for f in info["memory"].get_trainer_recv_funcs()]

        self.t0_diff_time = time.time()
        self.t0_q_act_to_mem = 0
        self.t0_q_mem_to_train = [0 for _ in range(len(self.mem_to_train_names))]
        self.t0_q_train_to_mem = 0

    def on_memory(self, context: RunContext, info: dict, **kwargs):
        if time.time() - self.progress_t0 > self.progress_timeout:
            self._print_memory(context, info)
            self._update_progress()
            self.progress_t0 = time.time()  # last

    def on_memory_end(self, context: RunContext, info: dict, **kwargs) -> None:
        self._print_memory(context, info)

    def _print_memory(self, context: RunContext, info: dict):
        _time = time.time()
        elapsed_time = _time - self.elapsed_t0
        diff_time = _time - self.t0_diff_time
        if diff_time < 0.1:
            diff_time = 0.1
        self.t0_diff_time = _time

        # [TIME] [actor] [elapsed time]
        s = datetime.datetime.now().strftime("%H:%M:%S")
        s += "  memory:"
        s += f" {to_str_time(elapsed_time)}"

        # [memory]
        mem = info["memory"].length()
        s += f" {mem:27d}mem"

        # [q]
        size = info["act_to_mem_size"]
        diff = info["act_to_mem"] - self.t0_q_act_to_mem
        s += f" {math.ceil(diff / diff_time):4d}recv/s [act->mem] {size:2d}q"
        self.t0_q_act_to_mem = info["act_to_mem"]

        size = info["train_to_mem_size"]
        diff = info["train_to_mem"] - self.t0_q_train_to_mem
        s += f" {math.ceil(diff / diff_time):4d}recv/s [train->mem] {size:2d}q"
        self.t0_q_train_to_mem = info["train_to_mem"]
        print(s)

        for i in range(len(info["mem_to_train_size_list"])):
            size = info["mem_to_train_size_list"][i]
            train = info["mem_to_train_list"][i]
            diff = train - self.t0_q_mem_to_train[i]
            s = " " * 55
            s += f" {math.ceil(diff / diff_time):4d}send/s [mem->train][{self.mem_to_train_names[i]}] {size:2d}q"
            self.t0_q_mem_to_train[i] = train
            print(s)
