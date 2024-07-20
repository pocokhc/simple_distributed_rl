import logging
import pprint
from typing import List, Optional, cast

from srl.base.context import RunContext
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import CallbackType, RunCallback, TrainCallback
from srl.utils import common

logger = logging.getLogger(__name__)


def play(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[CallbackType] = [],
    logger_config: bool = False,
):
    from srl.base.run import core_play

    context.check_stop_config()

    # --- callbacks ---
    callbacks_run: List[RunCallback] = cast(
        List[RunCallback],
        [c for c in callbacks if issubclass(c.__class__, RunCallback)],
    )
    [c.on_start(context) for c in callbacks_run]
    # -----------------

    try:
        # --- log ---
        if logger_config:
            logger.info("--- EnvConfig ---" + "\n" + pprint.pformat(context.env_config.to_dict()))
            logger.info("--- RLConfig ---" + "\n" + pprint.pformat(context.rl_config.to_dict()))
            logger.info(
                "--- Context ---"
                + "\n"
                + pprint.pformat(context.to_dict(include_env_config=False, include_rl_config=False))
            )
        # ------------

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
        [c.on_end(context) for c in callbacks_run]
        # -----------------

    return state


def play_generator(
    context: RunContext,
    env: EnvRun,
    workers: List[WorkerRun],
    main_worker_idx: int,
    trainer: Optional[RLTrainer] = None,
    callbacks: List[CallbackType] = [],
    logger_config: bool = False,
):
    from srl.base.run import core_play_generator

    context.check_stop_config()

    # --- callbacks ---
    callbacks_run: List[RunCallback] = cast(
        List[RunCallback],
        [c for c in callbacks if issubclass(c.__class__, RunCallback)],
    )
    [c.on_start(context) for c in callbacks_run]
    # -----------------

    try:
        # --- log ---
        if logger_config:
            logger.info("--- Context ---" + "\n" + pprint.pformat(context.to_dict()))
        # ------------

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
        [c.on_end(context) for c in callbacks_run]
        # -----------------
    return gen


def play_trainer_only(
    context: RunContext,
    trainer: RLTrainer,
    callbacks: List[CallbackType] = [],
    logger_config: bool = False,
):
    from srl.base.run import core_train_only

    context.check_stop_config()

    # --- callbacks ---
    callbacks_run: List[TrainCallback] = cast(
        List[TrainCallback],
        [c for c in callbacks if issubclass(c.__class__, TrainCallback)],
    )
    [c.on_start(context) for c in callbacks_run]
    # -----------------

    try:

        # --- log ---
        if logger_config:
            logger.info("--- Context ---" + "\n" + pprint.pformat(context.to_dict()))
            logger.info("--- Trainer ---" + "\n" + pprint.pformat(trainer.config.to_dict()))
        # ------------

        # --- random ---
        if context.seed is not None:
            common.set_seed(context.seed, context.seed_enable_gpu)
        # --------------

        state = core_train_only.play_trainer_only(context, trainer, callbacks_run)

    finally:
        # --- callbacks ---
        [c.on_end(context) for c in callbacks_run]
        # -----------------

    return state
