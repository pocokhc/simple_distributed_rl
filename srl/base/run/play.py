import logging
from typing import List, Optional, cast

from srl.base.context import RunContext
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import RunCallback
from srl.utils import common

logger = logging.getLogger(__name__)


def play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[RunCallback] = [],
):
    from srl.base.run import core_play

    context.check_stop_config()

    # --- callbacks ---
    callbacks_run: List[RunCallback] = cast(
        List[RunCallback],
        [c for c in callbacks if issubclass(c.__class__, RunCallback)],
    )
    [c.on_start(context=context) for c in callbacks_run]
    # -----------------

    try:
        logger.debug(context.to_str_context())

        # --- random ---
        if context.seed is not None:
            common.set_seed(context.seed, context.seed_enable_gpu)
        # --------------

        state = core_play.play(
            context,
            env,
            workers,
            main_worker_idx,
            trainer,
            callbacks_run,
        )
    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks_run]
        # -----------------

    return state


def play_generator(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[RunCallback] = [],
):
    from srl.base.run import core_play_generator

    context.check_stop_config()

    # --- callbacks ---
    callbacks_run: List[RunCallback] = cast(
        List[RunCallback],
        [c for c in callbacks if issubclass(c.__class__, RunCallback)],
    )
    [c.on_start(context=context) for c in callbacks_run]
    # -----------------

    try:
        logger.debug(context.to_str_context(include_env_config=False, include_rl_config=False))

        # --- random ---
        if context.seed is not None:
            common.set_seed(context.seed, context.seed_enable_gpu)
        # --------------

        gen = core_play_generator.play_generator(
            context,
            env,
            workers,
            main_worker_idx,
            trainer,
            callbacks_run,
        )
    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks_run]
        # -----------------
    return gen


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[RunCallback] = [],
):
    from srl.base.run import core_train_only

    context.check_stop_config()

    # --- callbacks ---
    [c.on_start(context=context) for c in callbacks]
    # -----------------

    try:
        logger.debug(context.to_str_context(include_env_config=False))

        # --- random ---
        if context.seed is not None:
            common.set_seed(context.seed, context.seed_enable_gpu)
        # --------------

        state = core_train_only.play_trainer_only(context, trainer, callbacks)

    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks]
        # -----------------

    return state
