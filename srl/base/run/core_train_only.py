import logging
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional

from srl.base.context import RunContext
from srl.base.rl.memory import IRLMemoryTrainer
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import FLAG_1STEP, RLTrainer
from srl.utils import common
from srl.utils.serialize import convert_for_json

from .callback import TrainCallback

logger = logging.getLogger(__name__)


@dataclass
class RunStateTrainer:
    trainer: RLTrainer
    memory: IRLMemoryTrainer
    parameter: RLParameter

    elapsed_t0: float = 0
    end_reason: str = ""

    # train
    is_step_trained: bool = False  # 非同期でタイミングを取るのに重要

    # thread
    thread_shared_dict: dict = field(default_factory=dict)
    thread_in_queue: Optional[queue.Queue] = None
    thread_out_queue: Optional[queue.Queue] = None

    # distributed
    sync_trainer: int = 0
    trainer_recv_q: int = 0

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
                if setup_data == FLAG_1STEP:
                    run_data = trainer.train()  # 互換用
                else:
                    run_data = trainer.train(setup_data)
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

    # --- 1 start
    state.trainer.on_start(context)

    # 2 callbacks
    _calls_on_train_before: List[Any] = [c for c in callbacks if hasattr(c, "on_train_before")]
    _calls_on_train_after: List[Any] = [c for c in callbacks if hasattr(c, "on_train_after")]
    [c.on_trainer_start(context, state) for c in callbacks]

    # --- thread
    if context.use_train_thread:
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
        t0_train_count = state.trainer.get_train_count()
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

            if context.max_train_count > 0 and state.trainer.get_train_count() >= context.max_train_count:
                state.end_reason = "max_train_count over."
                break

            # callbacks
            [c.on_train_before(context, state) for c in _calls_on_train_before]

            # --- train
            if context.use_train_thread:
                assert train_ps.is_alive()
                # Q send
                if state.thread_in_queue.qsize() < context.thread_queue_capacity:
                    setup_data = state.trainer.train_setup()
                    if setup_data is not None:
                        state.thread_in_queue.put(setup_data)
                else:
                    time.sleep(0.1)
                # Q recv
                if not state.thread_out_queue.empty():
                    for _ in range(state.thread_out_queue.qsize()):
                        run_data = state.thread_out_queue.get(timeout=1)
                        state.trainer.train_teardown(run_data)

                state.is_step_trained = state.trainer.get_train_count() > t0_train_count
                t0_train_count = state.trainer.get_train_count()
            else:
                state.is_step_trained = state.trainer.core_train()

            # callbacks
            _stop_flags = [c.on_train_after(context, state) for c in _calls_on_train_after]
            if True in _stop_flags:
                state.end_reason = "callback.trainer_intermediate_stop"
                break
    finally:
        if context.use_train_thread:
            state.thread_shared_dict["end_signal"] = True

    logger.info(f"loop end({state.end_reason})")

    # 4 end
    state.trainer.on_end()

    # 5 callbacks
    [c.on_trainer_end(context, state) for c in callbacks]
    return state
