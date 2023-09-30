import logging
import pickle
import pprint
import random
import time
from typing import List, Type, cast

from srl.base.rl.base import RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import Config, Context, Runner

logger = logging.getLogger(__name__)


class ShareBool:
    def __init__(self) -> None:
        self.val = False

    def get(self) -> bool:
        return self.val

    def set(self, val) -> None:
        self.val = val


# --------------------
# board
# --------------------
class Board:
    def __init__(self):
        self.params = None
        self.update_count = 0

    def write(self, params):
        self.params = params
        self.update_count += 1

    def get_update_count(self):
        return self.update_count

    def read(self):
        return self.params


# --------------------
# actor
# --------------------
def __run_actor(
    config: Config,
    context: Context,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ShareBool,
):
    # --- 関数をまたぐと yield を引き継ぐ必要があるので train を実装する必要あり
    try:
        context.run_name = f"actor{actor_id}"
        context.actor_id = actor_id
        runner = Runner(config.env_config, config.rl_config)
        runner.config = config
        runner.context = context
        logger.info(f"actor{actor_id} start.")

        # --- set_config_by_actor
        config.rl_config.set_config_by_actor(
            config.actor_num,
            context.actor_id,
        )

        # --- parameter
        parameter = runner.make_parameter()
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- train
        context.train_only = False
        context.disable_trainer = True
        context.training = True

        # -------------------------
        # yield にて制御する
        # -------------------------
        state = runner._create_play_state()
        state.remote_memory = remote_memory
        state.parameter = parameter

        # --- env/workers/trainer
        state.env = runner.make_env(is_init=True)
        state.workers = runner.make_players(parameter, remote_memory)

        # --- callbacks
        _callbacks = cast(List[Callback], [c for c in context.callbacks if issubclass(c.__class__, Callback)])
        [c.on_episodes_begin(runner) for c in _callbacks]

        # --- init
        state.elapsed_t0 = time.time()
        state.worker_indices = [i for i in range(state.env.player_num)]
        state.sync_actor = 0
        prev_update_count = 0

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
            if config.env_config.frameskip == 0:
                state.env.step(state.action)
            else:

                def __f():
                    [c.on_skip_step(runner) for c in _callbacks]

                state.env.step(state.action, __f)
            worker_idx = state.worker_indices[state.env.next_player_index]
            [w.on_step() for w in state.workers]
            state.total_step += 1

            # --- ActorInterrupt ---
            if state.total_step % config.actor_parameter_sync_interval_by_step == 0:
                update_count = remote_board.get_update_count()
                if update_count != prev_update_count:
                    prev_update_count = update_count
                    params = remote_board.read()
                    if params is not None:
                        parameter.restore(params)
                        state.sync_actor += 1
            # --- ActorInterrupt ---

            [c.on_step_end(runner) for c in _callbacks]
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

            if True in [c.intermediate_stop(runner) for c in _callbacks]:
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
def __run_trainer(
    config: Config,
    context: Context,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ShareBool,
):
    # --- 関数をまたぐと yield を引き継ぐ必要があるので train を実装する必要あり
    parameter = None
    try:
        context.run_name = "trainer"
        runner = Runner(config.env_config, config.rl_config)
        runner.config = config
        runner.context = context
        logger.info("trainer start.")

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- train
        context.train_only = True
        context.disable_trainer = False
        context.training = True

        # -------------------------
        # yield にて制御する
        # -------------------------
        state = runner._create_play_state()
        state.remote_memory = remote_memory
        state.parameter = parameter

        state.trainer = runner.make_trainer(parameter, remote_memory)
        _callbacks = cast(
            List[TrainerCallback], [c for c in context.callbacks if issubclass(c.__class__, TrainerCallback)]
        )
        [c.on_trainer_start(runner) for c in _callbacks]

        # --- init
        state.elapsed_t0 = time.time()

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
            state.train_info = state.trainer.train()
            [c.on_trainer_train(runner) for c in _callbacks]

            # --- TrainerInterrupt ---
            train_count = state.trainer.get_train_count()
            if train_count % config.trainer_parameter_send_interval_by_train_count == 0:
                remote_board.write(parameter.backup())
                runner.state.sync_trainer += 1
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


# ----------------------------
# 学習
# ----------------------------
def train(
    runner: Runner,
    save_remote_memory: str,
    return_remote_memory: bool,
    #
    choice_method: str = "random",
):
    logger.info(f"Config\n{pprint.pformat(runner.config.to_dict())}")
    logger.info(f"Context\n{pprint.pformat(runner.context.to_dict())}")

    logger.info("MPManager start")
    _train(runner, return_remote_memory, save_remote_memory, choice_method)
    logger.info("MPManager end")


def _train(
    runner: Runner,
    return_remote_memory: bool,
    save_remote_memory: str,
    choice_method: str,
):
    config = runner.config
    context = runner.context

    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- share values
    train_end_signal = ShareBool()
    remote_memory_class = cast(Type[RLRemoteMemory], make_remote_memory(config.rl_config, return_class=True))
    remote_memory = remote_memory_class(config.rl_config)
    remote_board = Board()

    # --- init remote_memory/parameter
    remote_memory.restore(runner.make_remote_memory().backup())
    remote_board.write(runner.make_parameter().backup())

    # --- actor
    actors_gen_list = []
    for actor_id in range(config.actor_num):
        actors_gen_list.append(
            __run_actor(
                pickle.loads(pickle.dumps(config)),
                pickle.loads(pickle.dumps(context)),
                remote_memory,
                remote_board,
                pickle.loads(pickle.dumps(actor_id)),
                train_end_signal,
            )
        )

    # --- trainer
    if context.disable_trainer:
        trainer_gen = None
    else:
        trainer_gen = __run_trainer(
            pickle.loads(pickle.dumps(config)),
            pickle.loads(pickle.dumps(context)),
            remote_memory,
            remote_board,
            train_end_signal,
        )

    # --- start
    logger.debug("process start")

    # callbacks
    [c.on_start(runner) for c in _callbacks]

    while True:
        if choice_method == "random":
            if random.random() < 0.8:
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

        # callbacks
        [c.on_polling(runner) for c in _callbacks]

        if train_end_signal.get():
            break
    logger.info("wait loop end.")

    # --- last parameter
    t0 = time.time()
    params = remote_board.read()
    if params is not None:
        runner._parameter = None
        runner.make_parameter().restore(params)
    logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

    # --- last memory
    if save_remote_memory != "":
        remote_memory.save(save_remote_memory, compress=True)
    if return_remote_memory:
        runner._remote_memory = None
        t0 = time.time()
        runner.make_remote_memory().restore(remote_memory.backup())
        logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s, len {runner.remote_memory.length()}")

    # callbacks
    [c.on_end(runner) for c in _callbacks]
