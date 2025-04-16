import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from srl.base.context import RunContext, RunNameTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import RunCallback
from srl.utils import common
from srl.utils.serialize import convert_for_json

logger = logging.getLogger(__name__)


@dataclass
class RunStateActor:
    env: EnvRun
    worker: WorkerRun  # main worker
    workers: List[WorkerRun]
    memory: RLMemory
    parameter: RLParameter
    trainer: Optional[RLTrainer]

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
    action: Any = None
    start_train_count: int = 0
    train_count: int = 0

    # train
    is_step_trained: bool = False

    # distributed
    sync_actor: int = 0
    actor_send_q: int = 0

    # info(簡単な情報はここに保存)
    last_episode_step: float = 0
    last_episode_time: float = 0
    last_episode_rewards: List[float] = field(default_factory=list)

    # other
    shared_vars: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat


def play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[RunCallback] = [],
):
    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            if context.run_name != RunNameTypes.eval:
                logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play(context, env, workers, main_worker_idx, trainer, callbacks)
    return _play(context, env, workers, main_worker_idx, trainer, callbacks)


def _play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer],
    callbacks: List[RunCallback],
):
    # --- check trainer
    if context.disable_trainer:
        trainer = None
    elif context.training:
        assert trainer is not None

    assert env.player_num == len(workers)
    main_worker = workers[main_worker_idx]
    state = RunStateActor(
        env,
        main_worker,
        workers,
        main_worker.worker.memory,
        main_worker.worker.parameter,
        trainer,
    )

    # --- 1 setup_from_actor
    if context.distributed:
        main_worker.config.setup_from_actor(context.actor_num, context.actor_id)

    # --- 2 random
    if context.seed is not None:
        state.episode_seed = random.randint(0, 2**16)
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- 3 setup
    state.env.setup(context)
    [w.setup(context) for w in state.workers]
    if state.trainer is not None:
        state.start_train_count = state.trainer.train_count
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
        if context.run_name != RunNameTypes.eval:
            logger.info(f"[{context.run_name}] loop start")
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
            # env render
            if context.rendering:
                state.env.render()

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
                state.train_count = state.trainer.train_count - state.start_train_count

            _stop_flags = [c.on_step_end(context=context, state=state) for c in _calls_on_step_end]
            state.worker_idx = state.worker_indices[state.env.next_player]  # on_step_end の後

            # ------------------------
            # done
            # ------------------------
            if state.env.done:
                # render
                if context.rendering:
                    state.env.render()

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
        if context.run_name != RunNameTypes.eval:
            logger.info(f"[{context.run_name}] loop end({state.end_reason})")

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

    return state
