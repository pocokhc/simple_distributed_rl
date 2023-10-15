import logging
import random
import time
from typing import List, Optional, cast

from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

from .callback import Callback, CallbackData, TrainerCallback
from .data import RunContext, RunState

logger = logging.getLogger(__name__)


def play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    trainer: RLTrainer,
    callback_data: Optional[CallbackData] = None,
) -> RunState:
    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play(context, env, workers, trainer, callback_data)
    return _play(context, env, workers, trainer, callback_data)


def _play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    trainer: RLTrainer,
    callback_data: Optional[CallbackData] = None,  # type: ignore , retype
) -> RunState:
    assert context._is_setup
    assert (
        context.max_steps > 0
        or context.max_episodes > 0
        or context.timeout > 0
        or context.max_train_count > 0
        or context.max_memory > 0
    ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count' or 'max_memory'."
    assert env.player_num == len(workers)

    state = RunState()
    state.env = env
    state.workers = workers
    state.memory = trainer.memory
    state.parameter = trainer.parameter
    if context.disable_trainer:
        state.trainer = None
    else:
        state.trainer = trainer
    if callback_data is None:
        callback_data: CallbackData = CallbackData()
    callback_data.set_data(context, state)

    # --- random
    if context.seed is not None:
        state.episode_seed = random.randint(0, 2**16)
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- callbacks
    _callbacks = cast(List[Callback], [c for c in context.callbacks if issubclass(c.__class__, Callback)])
    [c.on_episodes_begin(callback_data) for c in _callbacks]

    # --- init
    state.elapsed_t0 = time.time()
    state.worker_indices = [i for i in range(state.env.player_num)]

    def __skip_func():
        [c.on_skip_step(callback_data) for c in _callbacks]

    # --- loop
    logger.info(f"[{context.run_name}] loop start")
    env.init()
    while True:
        # --- stop check
        if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
            state.end_reason = "timeout."
            break

        if context.max_steps > 0 and state.total_step >= context.max_steps:
            state.end_reason = "max_steps over."
            break

        if state.trainer is not None:
            if context.max_train_count > 0 and state.trainer.get_train_count() >= context.max_train_count:
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
            state.env.reset(render_mode=context.render_mode, seed=state.episode_seed)

            if state.episode_seed is not None:
                state.episode_seed += 1

            # shuffle
            if context.shuffle_player:
                random.shuffle(state.worker_indices)
            state.worker_idx = state.worker_indices[state.env.next_player_index]

            # worker reset
            [
                w.on_reset(state.worker_indices[i], context.training, context.render_mode)
                for i, w in enumerate(state.workers)
            ]

            # callbacks
            [c.on_episode_begin(callback_data) for c in _callbacks]

        # ------------------------
        # step
        # ------------------------

        # action
        state.env.render()
        [c.on_step_action_before(callback_data) for c in _callbacks]
        state.action = state.workers[state.worker_idx].policy()

        # env step
        state.workers[state.worker_idx].render()
        [c.on_step_begin(callback_data) for c in _callbacks]
        state.env.step(state.action, __skip_func)
        worker_idx = state.worker_indices[state.env.next_player_index]

        # rl step
        [w.on_step() for w in state.workers]

        # step update
        state.total_step += 1

        # trainer
        if state.trainer is not None:
            state.trainer.train()

        [c.on_step_end(callback_data) for c in _callbacks]
        state.worker_idx = worker_idx

        if state.env.done:
            state.env.render()

            # rewardは学習中は不要
            if not context.training:
                worker_rewards = [
                    state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                ]
                state.episode_rewards_list.append(worker_rewards)

            [c.on_episode_end(callback_data) for c in _callbacks]

        # callback
        if True in [c.intermediate_stop(callback_data) for c in _callbacks]:
            state.end_reason = "callback.intermediate_stop"
            break
    logger.info(f"[{context.run_name}] loop end({state.end_reason})")

    # rewardは学習中は不要
    if not context.training:
        # 一度もepisodeを終了していない場合は例外で途中経過を保存
        if state.episode_count == 0:
            worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
            state.episode_rewards_list.append(worker_rewards)

    # callbacks
    [c.on_episodes_end(callback_data) for c in _callbacks]
    return state


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callback_data: Optional[CallbackData] = None,
) -> RunState:
    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play_trainer_only(context, trainer, callback_data)
    return _play_trainer_only(context, trainer, callback_data)


def _play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callback_data: Optional[CallbackData] = None,
):
    assert context._is_setup
    assert context.training
    assert context.max_train_count > 0 or context.timeout > 0, "Please specify 'max_train_count' or 'timeout'."

    state = RunState()
    state.trainer = trainer
    state.memory = trainer.memory
    state.parameter = trainer.parameter
    if callback_data is None:
        callback_data = CallbackData()
    callback_data.set_data(context, state)

    # --- callbacks
    _callbacks = cast(
        List[TrainerCallback], [c for c in context.callbacks if issubclass(c.__class__, TrainerCallback)]
    )
    [c.on_trainer_start(callback_data) for c in _callbacks]

    # --- init
    state.elapsed_t0 = time.time()

    # --- loop
    logger.info("loop start")
    while True:
        _time = time.time()

        # --- stop check
        if context.timeout > 0 and (_time - state.elapsed_t0) >= context.timeout:
            state.end_reason = "timeout."
            break

        if context.max_train_count > 0 and state.trainer.get_train_count() >= context.max_train_count:
            state.end_reason = "max_train_count over."
            break

        # --- train
        state.trainer.train()

        # callbacks
        [c.on_trainer_train(callback_data) for c in _callbacks]

        # callback end
        if True in [c.intermediate_stop(callback_data) for c in _callbacks]:
            state.end_reason = "callback.intermediate_stop"
            break
    logger.info(f"loop end({state.end_reason})")

    # callbacks
    [c.on_trainer_end(callback_data) for c in _callbacks]
    return state
