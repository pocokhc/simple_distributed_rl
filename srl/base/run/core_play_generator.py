import logging
import random
import time
from typing import Any, Generator, List, Optional, Tuple

from srl.base.context import RunContext, RunStateActor
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

logger = logging.getLogger(__name__)


def play_generator(
    context: RunContext,
    env: EnvRun,
    worker: WorkerRun,
    trainer: Optional[RLTrainer] = None,
    workers: Optional[List[WorkerRun]] = None,
) -> Generator[Tuple[str, RunContext, RunStateActor], None, None]:
    # Generator[YieldType, SendType, ReturnType]

    # --- context
    context.check_context_parameter(check_stop_config=False)  # generatorはチェックしない
    context.setup_memory_limit()
    context.setup_device()
    callbacks = context.callbacks

    # --- 0 check instance
    state = RunStateActor()
    state.env = env
    state.worker = worker
    state.parameter = worker.worker.parameter
    state.memory = worker.worker.memory

    if workers is None:
        workers, main_worker_idx = context.rl_config.make_workers(context.players, env, state.parameter, state.memory, worker)
    state.workers = workers

    if context.disable_trainer:
        trainer = None
    elif context.training and (trainer is None):
        trainer = context.rl_config.make_trainer(state.parameter, state.memory)
    state.trainer = trainer

    assert env.player_num == len(workers)

    # --- callbacks ---
    if not context.distributed:
        [c.on_start(context=context, state=state) for c in callbacks]
    # -----------------

    # --- 1 setup_from_actor
    if context.distributed:
        worker.config.setup_from_actor(context.actor_num, context.actor_id)

    # --- 2 random
    if context.seed is not None:
        common.set_seed(context.seed, context.seed_enable_gpu)

        state.episode_seed = random.randint(0, 2 ** (16 - 4))
        logger.info(f"set_seed: {context.seed}, 1st episode seed: {state.episode_seed}")

    # --- 3 setup
    env.setup(context, "" if context.rl_config is None else context.rl_config.request_env_render)
    [w.setup(context) for w in workers]
    if trainer is not None:
        trainer.setup(context)

    # --- 4 init
    state.worker_indices = [i for i in range(env.player_num)]

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
    if context.run_name != "eval":
        logger.debug(f"[{context.run_name}] loop start")
    state.elapsed_t0 = time.time()
    while True:
        # --- stop check
        if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
            state.end_reason = "timeout."
            break

        if context.max_steps > 0 and state.total_step >= context.max_steps:
            state.end_reason = "max_steps over."
            break

        if trainer is not None:
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
        if env.done:
            state.episode_count += 1
            if context.max_episodes > 0 and state.episode_count >= context.max_episodes:
                state.end_reason = "episode_count over."
                break  # end

            # env reset
            env.reset(seed=state.episode_seed)

            if state.episode_seed is not None:
                state.episode_seed += 1

            # shuffle
            if context.shuffle_player:
                random.shuffle(state.worker_indices)
            state.worker_idx = state.worker_indices[env.next_player]

            # worker reset
            [w.reset(state.worker_indices[i]) for i, w in enumerate(workers)]

            # callbacks
            yield ("on_episode_begin", context, state)
            [c.on_episode_begin(context=context, state=state) for c in _calls_on_episode_begin]

        # ------------------------
        # step
        # ------------------------
        yield ("on_step_begin", context, state)
        [c.on_step_begin(context=context, state=state) for c in _calls_on_step_begin]

        # --- action
        yield ("on_step_action_before", context, state)
        [c.on_step_action_before(context=context, state=state) for c in _calls_on_step_action_before]
        state.action = workers[state.worker_idx].policy()
        yield ("on_step_action_after", context, state)
        [c.on_step_action_after(context=context, state=state) for c in _calls_on_step_action_after]

        # workerがenvを終了させた場合に対応
        if not env.done:
            # env step
            env.step(
                state.action,
                workers[state.worker_idx].config.frameskip,
                __skip_func_arg,
            )

            # rl step
            [w.on_step() for w in workers]

            # step update
            state.total_step += 1

        # --- trainer
        if (trainer is not None) and (state.total_step % context.train_interval == 0):
            _prev_train = trainer.train_count
            for _ in range(context.train_repeat):
                trainer.train()
            state.is_step_trained = trainer.train_count > _prev_train
            if state.is_step_trained:
                state.train_count += trainer.train_count - _prev_train

        yield ("on_step_end", context, state)
        _stop_flags = [c.on_step_end(context=context, state=state) for c in _calls_on_step_end]
        state.worker_idx = state.worker_indices[env.next_player]  # on_step_end の後

        # ------------------------
        # done
        # ------------------------
        if env.done:
            # reward
            worker_rewards = [env.episode_rewards[state.worker_indices[i]] for i in range(env.player_num)]
            state.episode_rewards_list.append(worker_rewards)

            state.last_episode_step = env.step_num
            state.last_episode_time = env.elapsed_time
            state.last_episode_rewards = worker_rewards
            yield ("on_episode_end", context, state)
            [c.on_episode_end(context=context, state=state) for c in _calls_on_episode_end]

        if True in _stop_flags:
            state.end_reason = "callback.intermediate_stop"
            break

    if context.run_name != "eval":
        logger.debug(f"[{context.run_name}] loop end({state.end_reason})")

    # --- 7 teardown
    env.teardown()
    [w.teardown() for w in workers]
    if trainer is not None:
        trainer.teardown()

    # 一度もepisodeを終了していない場合は例外で途中経過を保存
    if state.episode_count == 0:
        worker_rewards = [env.episode_rewards[state.worker_indices[i]] for i in range(env.player_num)]
        state.episode_rewards_list.append(worker_rewards)
        state.last_episode_step = env.step_num
        state.last_episode_time = env.elapsed_time
        state.last_episode_rewards = worker_rewards

    # 8 callbacks
    yield ("on_episodes_end", context, state)
    [c.on_episodes_end(context=context, state=state) for c in callbacks]
    if not context.distributed:
        [c.on_end(context=context, state=state) for c in callbacks]
