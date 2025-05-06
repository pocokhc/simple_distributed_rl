import logging
import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, cast

from srl.base.context import RunContext, RunState
from srl.base.env.env_run import EnvRun
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import RunCallback
from srl.utils import common

logger = logging.getLogger(__name__)


@dataclass
class RunStateActor(RunState):
    env: EnvRun
    worker: WorkerRun
    workers: List[WorkerRun]
    memory: RLMemory
    parameter: RLParameter
    trainer: Optional["RLTrainer"]


def play(
    context: RunContext,
    state: Optional[RunState] = None,
    parameter_dat: Optional[Any] = None,
    memory_dat: Optional[Any] = None,
    callbacks: List[RunCallback] = [],
):
    # check context
    logger.debug(context.to_str_context())
    context.check_stop_config()

    if state is None:
        state = RunState()
    state.init()

    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            if context.run_name != "eval":
                logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play(context, cast(RunStateActor, state), parameter_dat, memory_dat, callbacks)
    return _play(context, cast(RunStateActor, state), parameter_dat, memory_dat, callbacks)


def _play(
    context: RunContext,
    state: RunStateActor,
    parameter_dat: Optional[Any],
    memory_dat: Optional[Any],
    callbacks: List[RunCallback],
):
    # --- 0 create instance
    if state.env is None:
        state.env = context.env_config.make()
    if state.parameter is None:
        state.parameter = context.rl_config.make_parameter(state.env)
    if parameter_dat is not None:
        state.parameter.restore(parameter_dat)
    if state.memory is None:
        state.memory = context.rl_config.make_memory(state.env)
    if memory_dat is not None:
        state.memory.restore(memory_dat)
    if (state.trainer is None) and context.training:
        state.trainer = context.rl_config.make_trainer(state.parameter, state.memory, state.env)
    if state.worker is None:
        state.worker = context.rl_config.make_worker(state.env, state.parameter, state.memory)
    if state.workers is None:
        state.workers, main_worker_idx = context.rl_config.make_workers(context.players, state.env, state.parameter, state.memory, state.worker)

    # check
    if context.disable_trainer:
        state.trainer = None
    elif context.training:
        assert state.trainer is not None
    assert state.env.player_num == len(state.workers)
    assert state.worker is not None

    # --- callbacks ---
    if not context.distributed:
        [c.on_start(context=context, state=state) for c in callbacks]
    # -----------------

    # --- 1 setup_from_actor
    if context.distributed:
        state.worker.config.setup_from_actor(context.actor_num, context.actor_id)

    # --- 2 random
    if context.seed is not None:
        common.set_seed(context.seed, context.seed_enable_gpu)

        state.episode_seed = random.randint(0, 2 ** (16 - 4))
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- 3 setup
    state.env.setup(context, "" if context.rl_config is None else context.rl_config.request_env_render)
    [w.setup(context) for w in state.workers]
    if state.trainer is not None:
        state.trainer.setup(context)

    # --- 4 init
    state.worker_indices = [i for i in range(state.env.player_num)]

    # --- 5 callbacks
    _calls_on_episode_begin: List[Any] = [c for c in callbacks if hasattr(c, "on_episode_begin")]
    _calls_on_episode_end: List[Any] = [c for c in callbacks if hasattr(c, "on_episode_end")]
    _calls_on_step_action_before: List[Any] = [c for c in callbacks if hasattr(c, "on_step_action_before")]
    _calls_on_step_action_after: List[Any] = [c for c in callbacks if hasattr(c, "on_step_action_after")]
    _calls_on_step_begin: List[Any] = [c for c in callbacks if hasattr(c, "on_step_begin")]
    _calls_on_step_end: List[Any] = [c for c in callbacks if hasattr(c, "on_step_end")]
    _calls_on_skip_step: List[Any] = [c for c in callbacks if hasattr(c, "on_skip_step")]
    if len(_calls_on_skip_step) > 0:

        def __skip_func():
            [c.on_skip_step(context=context, state=state) for c in _calls_on_skip_step]

        __skip_func_arg = __skip_func
    else:
        __skip_func_arg = None

    [c.on_episodes_begin(context=context, state=state) for c in callbacks]

    # --- 6 loop
    try:
        if context.run_name != "eval":
            logger.debug(f"[{context.run_name}] loop start")
        state.elapsed_t0 = time.time()
        while True:
            # --- stop check
            if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
                state.end_reason = "timeout."
                break

            if context.max_steps > 0 and state.total_step >= context.max_steps:
                state.end_reason = "max_steps over."
                break

            if state.trainer is not None:
                if context.max_train_count > 0 and state.train_count >= context.max_train_count:
                    state.end_reason = "max_train_count over."
                    break

            if state.memory is not None:
                if context.max_memory > 0 and state.memory.length() >= context.max_memory:
                    state.end_reason = "max_memory over."
                    break

            # ------------------------
            # episode end / init
            # ------------------------
            if state.env.done:
                state.episode_count += 1
                if context.max_episodes > 0 and state.episode_count >= context.max_episodes:
                    state.end_reason = "episode_count over."
                    break  # end

                # env reset
                state.env.reset(seed=state.episode_seed)

                if state.episode_seed is not None:
                    state.episode_seed += 1

                # shuffle
                if context.shuffle_player:
                    random.shuffle(state.worker_indices)
                state.worker_idx = state.worker_indices[state.env.next_player]

                # worker reset
                [w.reset(state.worker_indices[i], seed=state.episode_seed) for i, w in enumerate(state.workers)]

                # callbacks
                [c.on_episode_begin(context=context, state=state) for c in _calls_on_episode_begin]

            # ------------------------
            # step
            # ------------------------
            [c.on_step_begin(context=context, state=state) for c in _calls_on_step_begin]

            # --- action
            [c.on_step_action_before(context=context, state=state) for c in _calls_on_step_action_before]
            state.action = state.workers[state.worker_idx].policy()
            [c.on_step_action_after(context=context, state=state) for c in _calls_on_step_action_after]

            # workerがenvを終了させた場合に対応
            if not state.env.done:
                # env step
                state.env.step(
                    state.action,
                    state.workers[state.worker_idx].config.frameskip,
                    __skip_func_arg,
                )

                # rl step
                [w.on_step() for w in state.workers]

                # step update
                state.total_step += 1

            # --- trainer
            if state.trainer is not None:
                _prev_train = state.trainer.train_count
                state.trainer.train()
                state.is_step_trained = state.trainer.train_count > _prev_train
                if state.is_step_trained:
                    # 増えた分だけ加算
                    state.train_count += state.trainer.train_count - _prev_train

            _stop_flags = [c.on_step_end(context=context, state=state) for c in _calls_on_step_end]
            state.worker_idx = state.worker_indices[state.env.next_player]  # on_step_end の後

            # ------------------------
            # done
            # ------------------------
            if state.env.done:
                # reward
                worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
                state.episode_rewards_list.append(worker_rewards)

                state.last_episode_step = state.env.step_num
                state.last_episode_time = state.env.elapsed_time
                state.last_episode_rewards = worker_rewards
                [c.on_episode_end(context=context, state=state) for c in _calls_on_episode_end]

            if True in _stop_flags:
                state.end_reason = "callback.intermediate_stop"
                break
    finally:
        if context.run_name != "eval":
            logger.debug(f"[{context.run_name}] loop end({state.end_reason})")

        # --- 7 teardown
        state.env.teardown()
        [w.teardown() for w in state.workers]
        if state.trainer is not None:
            state.trainer.teardown()

        # rewardは学習中は不要
        if not context.training:
            # 一度もepisodeを終了していない場合は例外で途中経過を保存
            if state.episode_count == 0:
                worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
                state.episode_rewards_list.append(worker_rewards)
                state.last_episode_step = state.env.step_num
                state.last_episode_time = state.env.elapsed_time
                state.last_episode_rewards = worker_rewards

        # 8 callbacks
        [c.on_episodes_end(context=context, state=state) for c in callbacks]
        if not context.distributed:
            [c.on_end(context=context, state=state) for c in callbacks]

    return state
