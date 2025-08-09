import logging
import time
from typing import Any, List, Optional

from srl.base.context import RunContext, RunStateTrainer
from srl.base.rl.trainer import RLTrainer
from srl.utils import common

logger = logging.getLogger(__name__)


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    state: Optional[RunStateTrainer] = None,  # 継続用
    _check_tf: bool = True,
):
    if _check_tf and context.enable_tf_device and (context.framework == "tensorflow") and common.is_enable_tf_device_name(context.used_device_tf):
        import tensorflow as tf

        logger.info(f"tf.device({context.used_device_tf})")
        with tf.device(context.used_device_tf):  # type: ignore
            return play_trainer_only(
                context,
                trainer,
                state,
                _check_tf=False,
            )

    # --- context
    assert context.training
    context.setup()
    callbacks = context.callbacks

    # --- 0 check instance
    if state is None:
        state = RunStateTrainer()
    state.trainer = trainer
    state.parameter = trainer.parameter
    state.memory = trainer.memory

    # --- callbacks ---
    if not context.distributed:
        [c.on_start(context=context, state=state) for c in callbacks]
    # -----------------

    # --- 1 setup
    trainer.setup(context)

    # 2 callbacks
    _calls_on_train_before: List[Any] = [c for c in callbacks if hasattr(c, "on_train_before")]
    _calls_on_train_after: List[Any] = [c for c in callbacks if hasattr(c, "on_train_after")]
    [c.on_trainer_start(context=context, state=state) for c in callbacks]

    # --- 3 loop
    try:
        logger.debug("loop start")
        state.elapsed_t0 = time.time()
        while True:
            # --- stop check
            if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
                state.end_reason = "timeout."
                break

            if context.max_train_count > 0 and state.train_count >= context.max_train_count:
                state.end_reason = "max_train_count over."
                break

            # callbacks
            [c.on_train_before(context=context, state=state) for c in _calls_on_train_before]

            # --- train
            _prev_train = trainer.train_count
            trainer.train()
            state.is_step_trained = trainer.train_count > _prev_train
            if state.is_step_trained:
                # 増えた分だけ加算
                state.train_count += trainer.train_count - _prev_train

            # callbacks
            _stop_flags = [c.on_train_after(context=context, state=state) for c in _calls_on_train_after]
            if True in _stop_flags:
                state.end_reason = "callback.trainer_intermediate_stop"
                break
    finally:
        logger.debug(f"loop end({state.end_reason})")

        # 4 teardown
        trainer.teardown()

        # 5 callbacks
        [c.on_trainer_end(context=context, state=state) for c in callbacks]
        if not context.distributed:
            [c.on_end(context=context, state=state) for c in callbacks]

    return state
