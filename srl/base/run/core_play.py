import logging
import queue
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, cast

from srl.base.context import RunContext, RunNameTypes
from srl.base.define import EnvActionType
from srl.base.env.env_run import EnvRun
from srl.base.rl.memory import IRLMemoryWorker
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
    memory: IRLMemoryWorker
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
    action: EnvActionType = 0
    train_count: int = 0

    # train
    is_step_trained: bool = False

    # thread
    enable_train_thread: bool = False
    thread_shared_dict: dict = field(default_factory=dict)
    thread_in_queue: Optional[queue.Queue] = None
    thread_out_queue: Optional[queue.Queue] = None

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


def _train_thread(
    trainer: RLTrainer,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    thread_shared_dict: dict,
    thread_queue_capacity: int,
):
    try:
        while not thread_shared_dict["end_signal"]:
            if in_queue.empty() or (out_queue.qsize() >= thread_queue_capacity):
                time.sleep(0.1)
            else:
                setup_data = in_queue.get(timeout=1)
                run_data = trainer.thread_train(setup_data)
                out_queue.put(run_data)
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        logger.info("[TrainerThread] MemoryError")
    except Exception:
        logger.error(traceback.format_exc())
        logger.info("[TrainerThread] error")
    finally:
        logger.info("[TrainerThread] end")


def play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[RunCallback] = [],
):
    # --- check trainer
    if context.disable_trainer:
        trainer = None
    elif context.training:
        assert trainer is not None

    # --- play tf
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
    if context.enable_train_thread and (state.trainer is not None) and state.trainer.implement_thread_train():
        state.enable_train_thread = True

    # --- 1 setup_from_actor
    if context.distributed:
        main_worker.config.setup_from_actor(context.actor_num, context.actor_id)

    # --- 2 random
    if context.seed is not None:
        state.episode_seed = random.randint(0, 2**16)
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- 3 start
    [w.on_start(context) for w in state.workers]
    if state.trainer is not None:
        state.trainer.on_start(context)
    state.env.setup(context)

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
            [c.on_skip_step(context, state) for c in _calls_on_skip_step]

        __skip_func_arg = __skip_func
    else:
        __skip_func_arg = None

    [c.on_episodes_begin(context, state) for c in callbacks]

    # --- thread train
    if state.enable_train_thread:
        import threading

        state.thread_shared_dict = {"end_signal": False}
        state.thread_in_queue = queue.Queue()
        state.thread_out_queue = queue.Queue()
        train_ps = threading.Thread(
            target=_train_thread,
            args=(
                state.trainer,
                state.thread_in_queue,
                state.thread_out_queue,
                state.thread_shared_dict,
                context.thread_queue_capacity,
            ),
        )
        logger.info(f"[{context.run_name}] train thread start")
        t0_train_count = cast(RLTrainer, state.trainer).get_train_count()
        train_ps.start()

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
                state.env.reset(seed=state.episode_seed)

                if state.episode_seed is not None:
                    state.episode_seed += 1

                # shuffle
                if context.shuffle_player:
                    random.shuffle(state.worker_indices)
                state.worker_idx = state.worker_indices[state.env.next_player_index]

                # worker reset
                [w.on_reset(state.worker_indices[i], seed=state.episode_seed) for i, w in enumerate(state.workers)]

                # callbacks
                [c.on_episode_begin(context, state) for c in _calls_on_episode_begin]

            # ------------------------
            # step
            # ------------------------

            # action
            state.env.render()
            [c.on_step_action_before(context, state) for c in _calls_on_step_action_before]
            state.action = state.workers[state.worker_idx].policy()
            state.workers[state.worker_idx].render()
            [c.on_step_action_after(context, state) for c in _calls_on_step_action_after]

            # workerがenvを終了させた場合に対応
            if not state.env.done:
                # env step
                [c.on_step_begin(context, state) for c in _calls_on_step_begin]
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
                if state.enable_train_thread:
                    # Q send
                    if cast(queue.Queue, state.thread_in_queue).qsize() < context.thread_queue_capacity:
                        setup_data = state.trainer.thread_train_setup()
                        if setup_data is not None:
                            cast(queue.Queue, state.thread_in_queue).put(setup_data)
                    # Q recv
                    if not cast(queue.Queue, state.thread_out_queue).empty():
                        for _ in range(cast(queue.Queue, state.thread_out_queue).qsize()):
                            run_data = cast(queue.Queue, state.thread_out_queue).get(timeout=1)
                            state.trainer.thread_train_teardown(run_data)

                    state.is_step_trained = state.trainer.train_count > t0_train_count
                    t0_train_count = state.trainer.train_count
                elif not state.trainer.implement_thread_train():
                    _prev_train = state.trainer.train_count
                    state.trainer.train()
                    state.is_step_trained = state.trainer.train_count > _prev_train
                else:
                    setup_data = state.trainer.thread_train_setup()
                    if setup_data is not None:
                        _prev_train = state.trainer.train_count
                        train_data = state.trainer.thread_train(setup_data)
                        state.trainer.thread_train_teardown(train_data)
                        state.is_step_trained = state.trainer.train_count > _prev_train
                state.train_count = state.trainer.train_count

            _stop_flags = [c.on_step_end(context, state) for c in _calls_on_step_end]
            state.worker_idx = state.worker_indices[state.env.next_player_index]  # on_step_end の後

            if state.env.done:
                state.env.render()
                for w in state.workers:
                    if w.rendering:
                        # 最後のrender用policyは実施しない(RLWorker側で対処)
                        # try:
                        #    w.policy()
                        # except Exception:
                        #    logger.info(traceback.format_exc())
                        #    logger.info("Policy error in termination status (for rendering)")
                        w.render()

                # reward
                worker_rewards = [
                    state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                ]
                state.episode_rewards_list.append(worker_rewards)

                state.last_episode_step = state.env.step_num
                state.last_episode_time = state.env.elapsed_time
                state.last_episode_rewards = worker_rewards
                [c.on_episode_end(context, state) for c in _calls_on_episode_end]

            if True in _stop_flags:
                state.end_reason = "callback.intermediate_stop"
                break
    finally:
        if state.enable_train_thread:
            state.thread_shared_dict["end_signal"] = True

        if context.run_name != RunNameTypes.eval:
            logger.info(f"[{context.run_name}] loop end({state.end_reason})")

        # --- 7 end
        [w.on_end() for w in state.workers]
        if state.trainer is not None:
            state.trainer.on_end()

        # rewardは学習中は不要
        if not context.training:
            # 一度もepisodeを終了していない場合は例外で途中経過を保存
            if state.episode_count == 0:
                worker_rewards = [
                    state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                ]
                state.episode_rewards_list.append(worker_rewards)
                state.last_episode_step = state.env.step_num
                state.last_episode_time = state.env.elapsed_time
                state.last_episode_rewards = worker_rewards

        # 8 callbacks
        [c.on_episodes_end(context, state) for c in callbacks]

        if state.enable_train_thread:
            train_ps.join(timeout=10)

    return state
