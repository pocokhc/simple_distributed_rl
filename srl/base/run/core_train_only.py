import logging
import time
from dataclasses import dataclass, field
from typing import Any, List

from srl.base.context import RunContext
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.run.callback import RunCallback
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
    train_count: int = 0

    # train
    is_step_trained: bool = False  # 非同期でタイミングを取るのに重要

    # distributed
    sync_trainer: int = 0
    trainer_recv_q: int = 0

    # other
    shared_vars: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[RunCallback] = [],
):
    assert context.training

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
    callbacks: List[RunCallback],
):
    state = RunStateTrainer(trainer, trainer.memory, trainer.parameter)

    # --- 1 setup
    state.trainer.setup(context)

    # 2 callbacks
    _calls_on_train_before: List[Any] = [c for c in callbacks if hasattr(c, "on_train_before")]
    _calls_on_train_after: List[Any] = [c for c in callbacks if hasattr(c, "on_train_after")]
    [c.on_trainer_start(context=context, state=state) for c in callbacks]

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
            _prev_train = state.trainer.train_count
            state.trainer.train()
            state.is_step_trained = state.trainer.train_count > _prev_train
            if state.is_step_trained:
                # 増えた分だけ加算
                state.train_count += state.trainer.train_count - _prev_train

            # callbacks
            _stop_flags = [c.on_train_after(context=context, state=state) for c in _calls_on_train_after]
            if True in _stop_flags:
                state.end_reason = "callback.trainer_intermediate_stop"
                break
    finally:
        logger.info(f"loop end({state.end_reason})")

        # 4 teardown
        state.trainer.teardown()

        # 5 callbacks
        [c.on_trainer_end(context=context, state=state) for c in callbacks]

    return state
