import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from srl.base.define import EnvActionType
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import IRLMemoryTrainer, IRLMemoryWorker, RLMemory, RLParameter, RLTrainer
from srl.base.rl.registration import make_trainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.context import RunContext, RunNameTypes
from srl.utils import common
from srl.utils.serialize import convert_for_json

from .callback import RunCallback, TrainerCallback

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """
    実行中に変動する変数をまとめたクラス
    Class that summarizes variables that change during execution
    """

    env: Optional[EnvRun] = None
    workers: List[WorkerRun] = field(default_factory=list)
    trainer: Optional[RLTrainer] = None
    memory: Optional[Union[IRLMemoryWorker, IRLMemoryTrainer]] = None
    parameter: Optional[RLParameter] = None

    # episodes init
    elapsed_t0: float = 0
    worker_indices: List[int] = field(default_factory=list)

    # episode state
    episode_rewards_list: List[List[float]] = field(default_factory=list)
    episode_count: int = -1
    total_step: int = 0
    end_reason: str = ""
    worker_idx: int = 0
    episode_seed: Optional[int] = None
    action: EnvActionType = 0

    # train
    is_step_trained: bool = False

    # other
    sync_actor: int = 0
    sync_trainer: int = 0

    # ------------

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat


def play(
    context: RunContext,
    env: EnvRun,
    parameter: RLParameter,
    memory: RLMemory,
    workers: Optional[List[WorkerRun]] = None,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[RunCallback] = [],
) -> RunState:
    if not context.distributed:
        assert (
            context.max_steps > 0
            or context.max_episodes > 0
            or context.timeout > 0
            or context.max_train_count > 0
            or context.max_memory > 0
        ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count' or 'max_memory'."
        if context.max_memory > 0:
            if hasattr(memory, "config"):
                _m = getattr(memory.config, "memory", None)
                if _m is not None:
                    assert context.max_memory <= getattr(_m, "capacity", 0)

    # --- make instance
    if context.disable_trainer:
        trainer = None
    else:
        if trainer is None:
            trainer = make_trainer(context.rl_config, parameter, memory)
    if workers is None:
        controller = context.create_controller()
        workers = controller.make_workers(env, parameter, memory)

    # --- play tf
    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            if context.run_name != RunNameTypes.eval:
                logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play(context, env, parameter, memory, workers, trainer, callbacks)
    return _play(context, env, parameter, memory, workers, trainer, callbacks)


def _play(
    context: RunContext,
    env: EnvRun,
    parameter: RLParameter,
    memory: RLMemory,
    workers: List[WorkerRun],
    trainer: Optional[RLTrainer],
    callbacks: List[RunCallback] = [],
) -> RunState:
    assert env.player_num == len(workers)

    state = RunState()
    state.env = env
    state.parameter = parameter
    state.memory = memory
    state.workers = workers
    state.trainer = trainer

    # --- set_config_by_actor
    if context.distributed:
        context.rl_config.set_config_by_actor(context.actor_num, context.actor_id)

    # --- random
    if context.seed is not None:
        state.episode_seed = random.randint(0, 2**16)
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- callbacks
    [c.on_episodes_begin(context, state) for c in callbacks]

    # --- init
    state.elapsed_t0 = time.time()
    state.worker_indices = [i for i in range(state.env.player_num)]

    def __skip_func():
        [c.on_skip_step(context, state) for c in callbacks]

    # --- loop
    if context.run_name != RunNameTypes.eval:
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
            [c.on_episode_begin(context, state) for c in callbacks]

        # ------------------------
        # step
        # ------------------------

        # action
        state.env.render()
        [c.on_step_action_before(context, state) for c in callbacks]
        state.action = state.workers[state.worker_idx].policy()

        # env step
        state.workers[state.worker_idx].render()
        [c.on_step_begin(context, state) for c in callbacks]
        state.env.step(state.action, __skip_func)
        worker_idx = state.worker_indices[state.env.next_player_index]

        # rl step
        [w.on_step() for w in state.workers]

        # step update
        state.total_step += 1

        # trainer
        if state.trainer is not None:
            state.is_step_trained = state.trainer.train()

        _stop_flags = [c.on_step_end(context, state) for c in callbacks]
        state.worker_idx = worker_idx

        if state.env.done:
            state.env.render()

            # rewardは学習中は不要
            if not context.training:
                worker_rewards = [
                    state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                ]
                state.episode_rewards_list.append(worker_rewards)

            [c.on_episode_end(context, state) for c in callbacks]

        if True in _stop_flags:
            state.end_reason = "callback.intermediate_stop"
            break
    if context.run_name != RunNameTypes.eval:
        logger.info(f"[{context.run_name}] loop end({state.end_reason})")

    # rewardは学習中は不要
    if not context.training:
        # 一度もepisodeを終了していない場合は例外で途中経過を保存
        if state.episode_count == 0:
            worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
            state.episode_rewards_list.append(worker_rewards)

    # callbacks
    [c.on_episodes_end(context, state) for c in callbacks]
    return state


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[TrainerCallback] = [],
) -> RunState:
    assert context.training
    assert context.max_train_count > 0 or context.timeout > 0, "Please specify 'max_train_count' or 'timeout'."

    # --- play tf
    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play_trainer_only(context, trainer, callbacks)
    return _play_trainer_only(context, trainer, callbacks)


def _play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[TrainerCallback] = [],
):
    state = RunState()
    state.trainer = trainer
    state.parameter = trainer.parameter
    state.memory = trainer.memory

    # callbacks
    [c.on_trainer_start(context, state) for c in callbacks]

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
        state.is_step_trained = state.trainer.train()

        # callbacks
        _stop_flags = [c.on_trainer_loop(context, state) for c in callbacks]
        if True in _stop_flags:
            state.end_reason = "callback.trainer_intermediate_stop"
            break

    logger.info(f"loop end({state.end_reason})")

    # callbacks
    [c.on_trainer_end(context, state) for c in callbacks]
    return state
