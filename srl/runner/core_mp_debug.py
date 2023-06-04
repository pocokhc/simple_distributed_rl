import logging
import os
import pickle
import pprint
import random
import time
import traceback
from typing import List, Optional, Tuple, Type, cast

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.config import Config
from srl.runner.core import CheckpointOption, EvalOption, HistoryOption, Options, ProgressOption
from srl.utils.common import set_seed

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
    options: Options,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ShareBool,
):
    config._run_name = f"actor{actor_id}"
    config._actor_id = actor_id
    logger.info(f"actor{actor_id} start.")

    try:
        # --- config
        config._disable_trainer = True
        config.rl_config.set_config_by_actor(config.actor_num, actor_id)

        config.init_play()

        # --- parameter
        parameter = config.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- random seed
        episode_seed = None
        if config.seed is not None:
            set_seed(config.seed, config.seed_enable_gpu)
            episode_seed = random.randint(0, 2**16)
            logger.info(f"set_seed({config.seed})")
            logger.info(f"1st episode seed: {episode_seed}")

        # --- env/workers
        env = config.make_env()
        workers = config.make_players(parameter, remote_memory)

        # --- init
        _time = time.time()
        episode_count = -1
        total_step = 0
        elapsed_t0 = _time
        worker_indices = [i for i in range(env.player_num)]
        end_reason = ""
        worker_idx = 0

        # --- loop
        while True:
            yield

            _time = time.time()

            # --- stop check
            if config.timeout > 0 and (_time - elapsed_t0) > config.timeout:
                end_reason = "timeout."
                break

            if config.max_steps > 0 and total_step > config.max_steps:
                end_reason = "max_steps over."
                break

            # --- episode end / init
            if env.done:
                episode_count += 1

                if config.max_episodes > 0 and episode_count >= config.max_episodes:
                    end_reason = "episode_count over."
                    break

                # --- reset
                env.reset(seed=episode_seed)
                if episode_seed is not None:
                    episode_seed += 1

                # shuffle
                if config.shuffle_player:
                    random.shuffle(worker_indices)
                worker_idx = worker_indices[env.next_player_index]
                [w.on_reset(env, worker_indices[i]) for i, w in enumerate(workers)]

            # --- action
            action = workers[worker_idx].policy(env)

            # --- env step
            env.step(action)
            worker_idx = worker_indices[env.next_player_index]

            # --- worker step
            [w.on_step(env) for w in workers]

            total_step += 1

        logger.info(f"training end({end_reason})")

    finally:
        train_end_signal.set(True)
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
def __run_trainer(
    config: Config,
    options: Options,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ShareBool,
):
    config._run_name = "trainer"
    logger.info("trainer start.")

    # --- 関数をまたぐと yield を引き継ぐ必要があるので train を実装する必要あり
    # optionなどはTODO

    # --- parameter
    parameter = config.make_parameter(is_load=False)
    params = remote_board.read()
    if params is not None:
        parameter.restore(params)

    config.init_play()

    # --- random seed
    if config.seed is not None:
        set_seed(config.seed, config.seed_enable_gpu)
        logger.info(f"set_seed({config.seed})")

    # --- trainer
    trainer = config.make_trainer(parameter, remote_memory)

    _time = time.time()

    # --- init
    t0 = _time
    end_reason = ""

    try:
        while True:
            yield

            _time = time.time()

            # --- stop check
            if config.timeout > 0 and _time - t0 > config.timeout:
                end_reason = "timeout."
                break

            if config.max_train_count > 0 and trainer.get_train_count() > config.max_train_count:
                end_reason = "max_train_count over."
                break

            # --- train
            train_info = trainer.train()

        logger.info(f"training end({end_reason})")

    finally:
        train_end_signal.set(True)
        t0 = time.time()
        remote_board.write(parameter.backup())
        logger.info(f"trainer end.(send parameter time: {time.time() - t0:.1f}s)")


# ----------------------------
# 学習
# ----------------------------
def train(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
    disable_trainer: bool = False,
    enable_profiling: bool = True,
    # options
    eval: Optional[EvalOption] = None,
    progress: Optional[ProgressOption] = ProgressOption(),
    history: Optional[HistoryOption] = None,
    checkpoint: Optional[CheckpointOption] = None,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    return_remote_memory: bool = False,
    save_remote_memory: str = "",
    #
    choice_method: str = "random",
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    if disable_trainer:
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1
        ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."
    else:
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1 or max_train_count != -1
        ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."

    config.init_play()
    config = config.copy(env_share=False)

    # stop config
    config._max_episodes = max_episodes
    config._timeout = timeout
    config._max_steps = max_steps
    config._max_train_count = max_train_count
    # play config
    config._shuffle_player = shuffle_player
    config._disable_trainer = disable_trainer
    config._enable_profiling = enable_profiling
    # callbacks
    config._callbacks = callbacks[:]
    # play info
    config._training = True
    config._distributed = True

    options = Options()
    options.eval = eval
    options.progress = progress
    options.history = history
    options.checkpoint = checkpoint

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    logger.info(f"Training Config\n{pprint.pformat(config.to_dict())}")
    config.assert_params()

    logger.info("MPManager start")
    last_parameter, last_remote_memory = _train(
        config,
        options,
        parameter,
        remote_memory,
        disable_trainer,
        return_remote_memory,
        save_remote_memory,
        choice_method,
    )
    logger.info("MPManager end")

    # --- history
    return_history = HistoryViewer()
    try:
        if history is not None:
            return_history.load(config.save_dir)
    except Exception:
        logger.info(traceback.format_exc())

    return last_parameter, last_remote_memory, return_history


def _train(
    config: Config,
    options: Options,
    init_parameter: Optional[RLParameter],
    init_remote_memory: Optional[RLRemoteMemory],
    disable_trainer: bool,
    return_remote_memory: bool,
    save_remote_memory: str,
    choice_method,
):
    # callbacks
    _info = {
        "config": config,
    }
    [c.on_init(_info) for c in config.callbacks]

    # --- last params (errorチェックのため先に作っておく)
    last_parameter = config.make_parameter()
    last_remote_memory = config.make_remote_memory()

    # --- share values
    train_end_signal = ShareBool()
    remote_memory = cast(Type[RLRemoteMemory], make_remote_memory(config.rl_config, return_class=True))(
        config.rl_config
    )
    remote_board = Board()

    # init
    if init_remote_memory is None:
        if os.path.isfile(config.rl_config.remote_memory_path):
            remote_memory.load(config.rl_config.remote_memory_path)
    else:
        remote_memory.restore(init_remote_memory.backup())
    if init_parameter is None:
        _parameter = config.make_parameter()
        remote_board.write(_parameter.backup())
    else:
        remote_board.write(init_parameter.backup())

    # --- actor
    actors_gen_list = []
    for actor_id in range(config.actor_num):
        actors_gen_list.append(
            __run_actor(
                pickle.loads(pickle.dumps(config)),
                pickle.loads(pickle.dumps(options)),
                remote_memory,
                remote_board,
                pickle.loads(pickle.dumps(actor_id)),
                train_end_signal,
            )
        )

    # --- trainer
    if disable_trainer:
        trainer_gen = None
    else:
        trainer_gen = __run_trainer(
            pickle.loads(pickle.dumps(config)),
            pickle.loads(pickle.dumps(options)),
            remote_memory,
            remote_board,
            train_end_signal,
        )

    # --- start
    logger.debug("process start")

    # callbacks
    [c.on_start(_info) for c in config.callbacks]

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

        # callbacks
        [c.on_polling(_info) for c in config.callbacks]

        if train_end_signal.get():
            break
    logger.info("wait loop end.")

    # --- last parameter
    t0 = time.time()
    params = remote_board.read()
    if params is not None:
        last_parameter.restore(params)
    logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

    # --- last memory
    if save_remote_memory != "":
        remote_memory.save(save_remote_memory, compress=True)
    if return_remote_memory:
        t0 = time.time()
        last_remote_memory.restore(remote_memory.backup())
        logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s")

    # callbacks
    [c.on_end(_info) for c in config.callbacks]

    return last_parameter, last_remote_memory
