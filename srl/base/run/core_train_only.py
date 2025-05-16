import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, cast

from srl.base.context import RunContext, RunState
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.run.callback import RunCallback
from srl.utils import common

logger = logging.getLogger(__name__)


@dataclass
class RunStateTrainer(RunState):
    env: None
    worker: None
    workers: None
    trainer: RLTrainer
    memory: RLMemory
    parameter: RLParameter


def play_trainer_only(
    context: RunContext,
    state: Optional[RunState] = None,
    parameter_dat: Optional[Any] = None,
    memory_dat: Optional[Any] = None,
    callbacks: List[RunCallback] = [],
):
    # check context
    logger.debug(context.to_str_context())
    assert context.training
    context.check_stop_config()

    if state is None:
        state = RunState()
    state.init()

    if context.enable_tf_device and context.framework == "tensorflow":
        if common.is_enable_tf_device_name(context.used_device_tf):
            import tensorflow as tf

            logger.info(f"tf.device({context.used_device_tf})")
            with tf.device(context.used_device_tf):  # type: ignore
                return _play_trainer_only(context, cast(RunStateTrainer, state), parameter_dat, memory_dat, callbacks)
    return _play_trainer_only(context, cast(RunStateTrainer, state), parameter_dat, memory_dat, callbacks)


def _play_trainer_only(
    context: RunContext,
    state: RunStateTrainer,
    parameter_dat: Optional[Any],
    memory_dat: Optional[Any],
    callbacks: List[RunCallback],
):
    # --- 0 create instance
    if state.parameter is None:
        if state.trainer is None:
            state.parameter = context.rl_config.make_parameter(state.env)
        else:
            state.parameter = state.trainer.parameter
    if parameter_dat is not None:
        state.parameter.restore(parameter_dat)
    if state.memory is None:
        if state.trainer is None:
            state.memory = context.rl_config.make_memory(state.env)
        else:
            state.memory = state.trainer.memory
    if memory_dat is not None:
        state.memory.restore(memory_dat)
    if state.trainer is None:
        state.trainer = context.rl_config.make_trainer(state.parameter, state.memory, state.env)

    # --- callbacks ---
    if not context.distributed:
        [c.on_start(context=context, state=state) for c in callbacks]
    # -----------------

    # --- 1 setup
    state.trainer.setup(context)

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
        logger.debug(f"loop end({state.end_reason})")

        # 4 teardown
        state.trainer.teardown()

        # 5 callbacks
        [c.on_trainer_end(context=context, state=state) for c in callbacks]
        if not context.distributed:
            [c.on_end(context=context, state=state) for c in callbacks]

    return state
