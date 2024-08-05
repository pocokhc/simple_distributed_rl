import logging
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, cast

from srl.base.context import RunContext
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.run.callback import TrainCallback
from srl.utils import common
from srl.utils.serialize import convert_for_json

logger = logging.getLogger(__name__)


@dataclass
class RunStateTrainer:
    trainer: RLTrainer
    memory: RLMemory
    parameter: RLParameter

    elapsed_t0: float = 0
    end_reason: str = ""
    start_train_count: int = 0
    train_count: int = 0

    # train
    is_step_trained: bool = False  # 非同期でタイミングを取るのに重要

    # thread
    enable_train_thread: bool = False
    thread_shared_dict: dict = field(default_factory=dict)
    thread_in_queue: Optional[queue.Queue] = None
    thread_out_queue: Optional[queue.Queue] = None

    # distributed
    sync_trainer: int = 0
    trainer_recv_q: int = 0

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


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[TrainCallback] = [],
):
    assert context.training

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
    callbacks: List[TrainCallback],
):
    state = RunStateTrainer(trainer, trainer.memory, trainer.parameter)
    if context.enable_train_thread and trainer.implement_thread_train():
        state.enable_train_thread = True

    # --- 1 start
    state.start_train_count = state.trainer.train_count
    state.trainer.on_start(context)

    # 2 callbacks
    _calls_on_train_before: List[Any] = [c for c in callbacks if hasattr(c, "on_train_before")]
    _calls_on_train_after: List[Any] = [c for c in callbacks if hasattr(c, "on_train_after")]
    [c.on_trainer_start(context=context, state=state) for c in callbacks]

    # --- thread
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
        t0_train_count = state.trainer.train_count
        train_ps.start()

    # --- 3 loop
    try:
        logger.info("loop start")
        state.elapsed_t0 = time.time()
        while True:
            _time = time.time()

            # --- stop check
            if context.timeout > 0 and (_time - state.elapsed_t0) >= context.timeout:
                state.end_reason = "timeout."
                break

            if context.max_train_count > 0 and state.train_count >= context.max_train_count:
                state.end_reason = "max_train_count over."
                break

            # callbacks
            [c.on_train_before(context=context, state=state) for c in _calls_on_train_before]

            # --- train
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
            state.train_count = state.trainer.train_count - state.start_train_count

            # callbacks
            _stop_flags = [c.on_train_after(context=context, state=state) for c in _calls_on_train_after]
            if True in _stop_flags:
                state.end_reason = "callback.trainer_intermediate_stop"
                break
    finally:
        if state.enable_train_thread:
            state.thread_shared_dict["end_signal"] = True

    logger.info(f"loop end({state.end_reason})")

    # 4 end
    state.trainer.on_end()

    # 5 callbacks
    [c.on_trainer_end(context=context, state=state) for c in callbacks]
    return state
