import logging
import random
import time
from typing import List, Optional, cast

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.runner import Runner
from srl.utils import common

logger = logging.getLogger(__name__)


def play(runner: Runner, parameter: RLParameter, remote_memory: RLRemoteMemory):
    config = runner.config
    context = runner.context
    assert context._is_init

    # --- random seed ---
    if config.seed is not None:
        common.set_seed(config.seed, config.seed_enable_gpu)
        if context.run_name == "main":
            logger.info(f"set_seed({config.seed})")
    # -------------------

    # --- log context ---
    if context.training:
        import pprint

        if not context.distributed:
            logger.info(f"Config\n{pprint.pformat(config.to_dict())}")
        if context.actor_id == 0:
            logger.info(f"Context\n{pprint.pformat(context.to_dict())}")
    # -------------------

    # --- play(+tf) ----
    if (
        (not config.tf_device_disable)
        and (context.framework == "tensorflow")
        and common.is_enable_tf_device_name(context.used_device_tf)
    ):
        import tensorflow as tf

        if (not context.distributed) and (context.run_name == "main"):
            logger.info(f"tf.device({context.used_device_tf})")

        with tf.device(context.used_device_tf):  # type: ignore
            _play_main(runner, parameter, remote_memory)
    else:
        _play_main(runner, parameter, remote_memory)
    # ----------------


def _play_main(runner: Runner, parameter: RLParameter, remote_memory: RLRemoteMemory):
    if not runner.context.train_only:
        _play_run(runner, parameter, remote_memory)
    else:
        _play_train_only(runner, parameter, remote_memory)


def _play_run(runner: Runner, parameter: RLParameter, remote_memory: RLRemoteMemory):
    config = runner.config
    context = runner.context
    state = runner._create_play_state()
    state.remote_memory = remote_memory
    state.parameter = parameter

    # --- random seed ---
    if config.seed is not None:
        state.episode_seed = random.randint(0, 2**16)
        if context.run_name == "main":
            logger.info(f"1st episode seed: {state.episode_seed}")
    # -------------------

    # --- env/workers/trainer
    state.env = runner.make_env(is_init=True)
    state.workers = runner.make_players(parameter, remote_memory)
    if runner.context.training and not runner.context.disable_trainer:
        state.trainer = runner.make_trainer(parameter, remote_memory)

    # --- callbacks
    _callbacks = cast(List[Callback], [c for c in context.callbacks if issubclass(c.__class__, Callback)])
    [c.on_episodes_begin(runner) for c in _callbacks]

    # --- init
    state.elapsed_t0 = time.time()
    state.worker_indices = [i for i in range(state.env.player_num)]

    # --- loop
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
            [c.on_episode_begin(runner) for c in _callbacks]

        # ------------------------
        # step
        # ------------------------
        [c.on_step_action_before(runner) for c in _callbacks]

        # action
        state.action = state.workers[state.worker_idx].policy()

        # callbacks
        [c.on_step_begin(runner) for c in _callbacks]

        # env step
        if config.env_config.frameskip == 0:
            state.env.step(state.action)
        else:

            def __f():
                [c.on_skip_step(runner) for c in _callbacks]

            state.env.step(state.action, __f)
        worker_idx = state.worker_indices[state.env.next_player_index]

        # rl step
        [w.on_step() for w in state.workers]

        # step update
        state.total_step += 1

        # trainer
        if state.trainer is not None:
            state.train_info = state.trainer.train()

        [c.on_step_end(runner) for c in _callbacks]
        state.worker_idx = worker_idx

        if state.env.done:
            # rewardは学習中は不要
            if not context.training:
                worker_rewards = [
                    state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                ]
                state.episode_rewards_list.append(worker_rewards)

            [c.on_episode_end(runner) for c in _callbacks]

        # callback
        if True in [c.intermediate_stop(runner) for c in _callbacks]:
            state.end_reason = "callback.intermediate_stop"
            break

    if context.training:
        logger.info(f"training end({state.end_reason})")

    # rewardは学習中は不要
    if not context.training:
        # 一度もepisodeを終了していない場合は例外で途中経過を保存
        if state.episode_count == 0:
            worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
            state.episode_rewards_list.append(worker_rewards)

    # callbacks
    [c.on_episodes_end(runner) for c in _callbacks]


def _play_train_only(runner: Runner, parameter: Optional[RLParameter], remote_memory: Optional[RLRemoteMemory]):
    context = runner.context
    state = runner._create_play_state()
    state.remote_memory = remote_memory
    state.parameter = parameter

    # --- trainer
    state.trainer = runner.make_trainer(parameter, remote_memory)

    # --- callbacks
    _callbacks = cast(
        List[TrainerCallback], [c for c in context.callbacks if issubclass(c.__class__, TrainerCallback)]
    )
    [c.on_trainer_start(runner) for c in _callbacks]

    # --- init
    state.elapsed_t0 = time.time()

    # --- loop
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
        state.train_info = state.trainer.train()

        # callbacks
        [c.on_trainer_train(runner) for c in _callbacks]

        # callback end
        if True in [c.intermediate_stop(runner) for c in _callbacks]:
            state.end_reason = "callback.intermediate_stop"
            break

    # callbacks
    [c.on_trainer_end(runner) for c in _callbacks]
    logger.info(f"training end({state.end_reason})")
