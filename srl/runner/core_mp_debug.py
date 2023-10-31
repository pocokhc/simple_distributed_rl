import logging
import pickle
import random
import time
from queue import Queue
from typing import List, cast

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.base.run.core import RunState
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import TaskConfig

logger = logging.getLogger(__name__)


class _ShareBool:
    def __init__(self) -> None:
        self.val = False

    def get(self) -> bool:
        return self.val

    def set(self, val) -> None:
        self.val = val


# --------------------
# board
# --------------------
class _Board:
    def __init__(self):
        self.params = None

    def write(self, params):
        self.params = params

    def read(self):
        return self.params


# --------------------
# actor
# --------------------
class _RLMemory(IRLMemoryWorker):
    def __init__(self, memory_queue: Queue):
        self.queue = memory_queue

    def add(self, *args) -> None:
        self.queue.put(args)

    def length(self) -> int:
        return -1


def _run_actor(
    mp_data: TaskConfig,
    memory_queue: Queue,
    remote_board: _Board,
    actor_id: int,
    train_end_signal: _ShareBool,
):
    # --- 関数をまたぐと yield を引き継ぐ必要があるので train を実装する必要あり
    try:
        logger.info(f"actor{actor_id} start.")
        context = mp_data.context
        context.run_name = RunNameTypes.actor
        context.actor_id = actor_id

        # --- set_config_by_actor
        mp_data.context.rl_config.set_config_by_actor(context.actor_num, context.actor_id)

        # --- memory
        memory = cast(RLMemory, _RLMemory(memory_queue))

        # --- runner
        runner = srl.Runner(
            mp_data.context.env_config,
            mp_data.context.rl_config,
            mp_data.config,
            context,
            memory=memory,
        )

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- train
        runner.context.training = True
        runner.context.disable_trainer = True

        # -------------------------
        # yield にて制御する
        # -------------------------
        runner.state = state = RunState()
        state.memory = memory
        state.parameter = parameter

        # --- env/workers/trainer
        state.env = runner.make_env(is_init=True)
        state.workers = runner.make_workers(parameter, memory)

        # --- callbacks
        _callbacks = cast(List[Callback], [c for c in context.callbacks if issubclass(c.__class__, Callback)])
        [c.on_episodes_begin(runner) for c in _callbacks]

        # --- init
        state.elapsed_t0 = time.time()
        state.worker_indices = [i for i in range(state.env.player_num)]
        state.sync_actor = 0
        t0_sync = time.time()

        def __skip_func():
            [c.on_skip_step(runner) for c in _callbacks]

        # --- loop
        while True:
            yield

            # --- stop check
            if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
                state.end_reason = "timeout."
                break

            if context.max_steps > 0 and state.total_step >= context.max_steps:
                state.end_reason = "max_steps over."
                break

            # --- episode end / init
            if state.env.done:
                state.episode_count += 1

                if context.max_episodes > 0 and state.episode_count >= context.max_episodes:
                    state.end_reason = "episode_count over."
                    break  # end

                # --- reset
                state.env.reset(render_mode=context.render_mode)

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

            # --- step
            [c.on_step_action_before(runner) for c in _callbacks]
            state.action = state.workers[state.worker_idx].policy()
            [c.on_step_begin(runner) for c in _callbacks]

            # env step
            state.env.step(state.action, __skip_func)
            worker_idx = state.worker_indices[state.env.next_player_index]
            [w.on_step() for w in state.workers]
            state.total_step += 1

            # --- ActorInterrupt ---
            if time.time() - t0_sync > mp_data.config.actor_parameter_sync_interval:
                t0_sync = time.time()
                params = remote_board.read()
                if params is not None:
                    parameter.restore(params)
                    state.sync_actor += 1
            # --- ActorInterrupt ---

            _stop_flags = [c.on_step_end(runner) for c in _callbacks]
            state.worker_idx = worker_idx

            if state.env.done:
                if not context.training:
                    worker_rewards = [
                        state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)
                    ]
                    state.episode_rewards_list.append(worker_rewards)

                [c.on_episode_end(runner) for c in _callbacks]

            # --- ActorInterrupt ---
            if train_end_signal.get():
                state.end_reason = "train_end_signal"
                break
            # --- ActorInterrupt ---

            if True in _stop_flags:
                state.end_reason = "callback.intermediate_stop"
                break

        logger.info(f"training end({state.end_reason})")

        # 一度もepisodeを終了していない場合は例外で途中経過を保存
        if state.episode_count == 0:
            worker_rewards = [state.env.episode_rewards[state.worker_indices[i]] for i in range(state.env.player_num)]
            state.episode_rewards_list.append(worker_rewards)

        [c.on_episodes_end(runner) for c in _callbacks]

    finally:
        train_end_signal.set(True)
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
def _run_trainer(
    mp_data: TaskConfig,
    parameter: RLParameter,
    memory: RLMemory,
    remote_board: _Board,
    train_end_signal: _ShareBool,
):
    # --- 関数をまたぐと yield を引き継ぐ必要があるので train を実装する必要あり
    try:
        logger.info("trainer start.")

        # --- runner
        runner = srl.Runner(
            mp_data.context.env_config,
            mp_data.context.rl_config,
            mp_data.config,
            mp_data.context,
            parameter,
            memory,
        )

        # --- trainer
        trainer = runner.make_trainer()

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- train
        context = runner.context
        context.training = True

        # -------------------------
        # yield にて制御する
        # -------------------------
        state = RunState()
        state.memory = memory
        state.parameter = parameter
        state.trainer = trainer
        runner.state = state

        _callbacks = cast(
            List[TrainerCallback], [c for c in context.callbacks if issubclass(c.__class__, TrainerCallback)]
        )
        [c.on_trainer_start(runner) for c in _callbacks]

        # --- init
        state.elapsed_t0 = time.time()
        t0_sync = time.time()

        while True:
            yield

            # --- stop check
            if context.timeout > 0 and (time.time() - state.elapsed_t0) >= context.timeout:
                state.end_reason = "timeout."
                break

            if context.max_train_count > 0 and state.trainer.get_train_count() >= context.max_train_count:
                state.end_reason = "max_train_count over."
                break

            # --- train
            state.trainer.train()
            _stop_flags = [c.on_trainer_loop(runner) for c in _callbacks]

            # --- TrainerInterrupt ---
            if time.time() - t0_sync > mp_data.config.trainer_parameter_send_interval:
                t0_sync = time.time()
                remote_board.write(parameter.backup())
                runner.state.sync_trainer += 1

            if True in _stop_flags:
                state.end_reason = "callback"
                break

            if train_end_signal.get():
                state.end_reason = "train_end_signal"
                break
            # --- TrainerInterrupt ---

        [c.on_trainer_end(runner) for c in _callbacks]
        logger.info(f"training end({state.end_reason})")

    finally:
        train_end_signal.set(True)
        if parameter is not None:
            t0 = time.time()
            remote_board.write(parameter.backup())
            logger.info(f"trainer end.(send parameter time: {time.time() - t0:.1f}s)")


# --------------------
# memory
# --------------------
def _run_memory(
    memory: RLMemory,
    memory_queue: Queue,
    train_end_signal: _ShareBool,
):
    while not train_end_signal.get():
        yield
        if not memory_queue.empty():
            batch = memory_queue.get()
            memory.add(*batch)


# ----------------------------
# 学習
# ----------------------------
def train(runner: srl.Runner, choice_method: str = "random"):
    context = runner.context
    runner.make_env()

    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- share values
    train_end_signal = _ShareBool()
    memory_queue: Queue = Queue()
    remote_board = _Board()

    # --- init remote_memory/parameter
    parameter = runner.make_parameter()
    memory = runner.make_memory()
    remote_board.write(parameter.backup())

    # --- actor
    actors_gen_list = []
    for actor_id in range(context.actor_num):
        actors_gen_list.append(
            _run_actor(
                pickle.loads(pickle.dumps(runner.create_task_config(["Checkpoint"]))),
                memory_queue,
                remote_board,
                pickle.loads(pickle.dumps(actor_id)),
                train_end_signal,
            )
        )

    # --- trainer
    trainer_gen = _run_trainer(
        pickle.loads(pickle.dumps(runner.create_task_config())),
        parameter,
        memory,
        remote_board,
        train_end_signal,
    )

    # --- memory
    memory_gen = _run_memory(memory, memory_queue, train_end_signal)

    # --- start
    logger.debug("process start")

    # callbacks
    [c.on_start(runner) for c in _callbacks]

    while True:
        if choice_method == "random":
            r = random.random()
            if r < 0.5:
                next(memory_gen)
            elif r < 0.9:
                gen = random.choice(actors_gen_list)
                try:
                    next(gen)
                except StopIteration:
                    actors_gen_list.remove(gen)
            else:
                if trainer_gen is not None:
                    try:
                        next(trainer_gen)
                    except StopIteration:
                        trainer_gen = None
        elif choice_method == "ratio":
            raise NotImplementedError("TODO")
        else:
            raise ValueError(choice_method)

        if train_end_signal.get():
            break
    logger.info("wait loop end.")

    # --- last parameter
    t0 = time.time()
    params = remote_board.read()
    if params is not None:
        runner.parameter = None
        runner.make_parameter().restore(params)
    logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

    # callbacks
    [c.on_end(runner) for c in _callbacks]
