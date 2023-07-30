import logging
import multiprocessing as mp
import os
import pprint
import threading
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, Optional, Tuple

import numpy as np

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.utils.common import is_enable_tf_device_name

logger = logging.getLogger(__name__)


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


class ServerManager:
    def __init__(self, max_actor: int = 1) -> None:
        self.max_actor = max_actor
        self.actor_nums = [0 for _ in range(max_actor)]
        self.status: str = "WAIT"

    def is_running(self) -> bool:
        return self.status == "RUNNING"

    def is_end(self) -> bool:
        return self.status == "END"

    def start(self):
        self.status = "RUNNING"

    def set_end(self) -> None:
        self.status = "END"

    def join_actor(self) -> int:
        # まだ参加していないactor_idを返す
        min_index = int(np.argmin(self.actor_nums))
        self.actor_nums[min_index] += 1
        s = f"join actor: {min_index}"
        logger.info(s)
        print(s)
        return min_index

    def disconnected(self, actor_id: int) -> None:
        if not (0 <= actor_id < len(self.actor_nums)):
            logger.warning(f"out of index: {actor_id} (actors: {self.actor_nums})")
            return
        self.actor_nums[actor_id] -= 1
        if self.actor_nums[actor_id] < 0:
            logger.warning(f"invalid value: {actor_id} (actors: {self.actor_nums})")
            self.actor_nums[actor_id] = 0
        logger.info(f"disconnected actor: {actor_id}")

    def get_actors_count(self) -> int:
        # 1以上の数を返す
        return len([n for n in self.actor_nums if n >= 1])


class MPManager(BaseManager):
    pass


# --------------------
# actor
# --------------------
class __ActorInterrupt(Callback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        server_manager: ServerManager,
        config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.server_manager = server_manager
        self.actor_parameter_sync_interval_by_step = config.actor_parameter_sync_interval_by_step

        self.step = 0
        self.prev_update_count = 0
        self.sync_count = 0

    def on_episodes_begin(self, info):
        info["sync"] = 0

    def on_step_end(self, info):
        self.step += 1
        if self.step % self.actor_parameter_sync_interval_by_step != 0:
            return
        update_count = self.remote_board.get_update_count()
        if update_count == self.prev_update_count:
            return
        self.prev_update_count = update_count
        params = self.remote_board.read()
        if params is None:
            return
        self.parameter.restore(params)
        self.sync_count += 1
        info["sync"] = self.sync_count

    def intermediate_stop(self, info) -> bool:
        return self.server_manager.is_end()


def run_actor(
    server_ip: str,
    port: int,
    authkey: bytes = b"abracadabra",
    actor_id: Optional[int] = None,
    verbose: bool = True,
):
    MPManager.register("get_config")
    MPManager.register("get_options")
    MPManager.register("ServerManager")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")

    s = f"connect server: {server_ip}:{port}"
    logger.info(s)
    if verbose:
        print(s)
    manager: Any = MPManager(address=(server_ip, port), authkey=authkey)
    manager.connect()

    config: Config = manager.get_config().copy()
    options: Options = manager.get_options().copy()
    server_manager: ServerManager = manager.ServerManager()
    remote_board: Board = manager.Board()
    remote_memory = manager.RemoteMemory()

    logger.info(f"Config\n{pprint.pformat(config.to_dict())}")

    # --- env check
    try:
        config.make_env()
    except Exception as e:
        logger.debug(traceback.format_exc())
        s = f"Environment '{config.env_config.name}' could not be loaded.(System: {e})"
        s += "\nCheck the registration status of your environment.(Forgot to import environment?)"
        logger.info(s)
        print(s)
        return

    # --- join server
    actor_id = server_manager.join_actor()
    try:
        config._run_name = f"actor{actor_id}"
        config._actor_id = actor_id
        config.init_process()

        s = f"Joined server(Your actor index is {actor_id})"
        logger.info(s)
        if verbose:
            print(s)

        # --- config
        config._disable_trainer = True
        config._timeout = -1
        config.rl_config.set_config_by_actor(config.actor_num, actor_id)

        # --- parameter
        parameter = config.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        callbacks: List[Callback] = [
            __ActorInterrupt(
                remote_board,
                parameter,
                server_manager,
                config,
            )
        ]

        # --- wait
        print("Waiting for server to start...")
        while True:
            if server_manager.is_running():
                break
            time.sleep(1)  # polling

        # --- play
        allocate = config.used_device_tf
        if (not config.tf_disable) and is_enable_tf_device_name(allocate):
            import tensorflow as tf

            logger.info(f"actor{actor_id} start(allocate={allocate})")
            with tf.device(allocate):  # type: ignore
                __run_actor_main(config, options, parameter, remote_memory, callbacks)

        else:
            logger.info(f"actor{actor_id} start.")
            __run_actor_main(config, options, parameter, remote_memory, callbacks)

    finally:
        server_manager.disconnected(actor_id)


def __run_actor_main(
    config: Config,
    options: Options,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
    callbacks: List[Callback],
):
    core.play(
        config,
        # stop config
        max_episodes=config.max_episodes,
        timeout=config.timeout,
        max_steps=config.max_steps,
        max_train_count=config.max_train_count,
        # play config
        train_only=False,
        shuffle_player=config.shuffle_player,
        disable_trainer=True,
        enable_profiling=config.enable_profiling,
        # play info
        training=True,
        distributed=True,
        # option
        eval=options.eval,
        progress=options.progress,
        history=options.history,
        checkpoint=options.checkpoint,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )


# --------------------
# trainer
# --------------------
class __TrainerInterrupt(Callback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        server_manager: ServerManager,
        config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.server_manager = server_manager
        self.config = config

        self.sync_count = 0

    def on_trainer_start(self, info):
        info["sync"] = 0

    def on_trainer_train(self, info):
        train_count = info["train_count"]

        if train_count == 0:
            time.sleep(1)
            return

        if train_count % self.config.trainer_parameter_send_interval_by_train_count == 0:
            self.remote_board.write(self.parameter.backup())
            self.sync_count += 1
            info["sync"] = self.sync_count

    def intermediate_stop(self, info) -> bool:
        return self.server_manager.is_end()


def __run_trainer(
    last_param_queue: mp.Queue,
    last_remote_memory_queue: mp.Queue,
    config: Config,
    options: Options,
    save_remote_memory: bool,
):
    config._run_name = "trainer"
    config.init_process()

    MPManager.register("ServerManager")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")

    manager: Any = MPManager(address=("127.0.0.1", config.remote_port), authkey=config.remote_authkey)
    manager.connect()
    server_manager: ServerManager = manager.ServerManager()
    parameter = None
    remote_memory = None
    try:
        remote_board: Board = manager.Board()
        remote_memory = manager.RemoteMemory()

        parameter = config.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        callbacks: List[Callback] = [
            __TrainerInterrupt(
                remote_board,
                parameter,
                server_manager,
                config,
            )
        ]

        # --- play
        allocate = config.used_device_tf
        if (not config.tf_disable) and is_enable_tf_device_name(allocate):
            import tensorflow as tf

            logger.info(f"trainer start(allocate={allocate})")
            with tf.device(allocate):  # type: ignore
                __run_trainer_main(config, options, parameter, remote_memory, callbacks)
        else:
            logger.info("trainer start.")
            __run_trainer_main(config, options, parameter, remote_memory, callbacks)

    finally:
        try:
            server_manager.set_end()
            logger.info("trainer end.")
        finally:
            # --- send last params
            if parameter is not None:
                t0 = time.time()
                last_param_queue.put(parameter.backup())
                logger.info(f"send parameter time: {time.time() - t0:.1f}s")

            # --- memory
            if remote_memory is not None and save_remote_memory != "":
                remote_memory.save(save_remote_memory, compress=True)

            # --- send last memory
            if remote_memory is not None:
                t0 = time.time()
                last_remote_memory_queue.put(remote_memory.backup())
                logger.info(f"send remote_memory time: {time.time() - t0:.1f}s)")


def __run_trainer_main(
    config: Config,
    options: Options,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
    callbacks: List[Callback],
):
    core.play(
        config,
        # stop config
        max_episodes=-1,
        timeout=config.timeout,
        max_steps=-1,
        max_train_count=config.max_train_count,
        #  play config
        train_only=True,
        shuffle_player=False,
        disable_trainer=False,
        enable_profiling=config.enable_profiling,
        # play info
        training=True,
        distributed=True,
        # option
        eval=options.eval,
        progress=options.progress,
        history=options.history,
        checkpoint=options.checkpoint,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )


# --------------------
# server
# --------------------
def __run_server(
    config: Config,
    options: Options,
    init_parameter: RLParameter,
    init_remote_memory: RLRemoteMemory,
):
    # とりあえずCPUにしておく
    if config.use_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("[server] set CUDA_VISIBLE_DEVICES=-1")

    server_manager = ServerManager()
    board = Board()
    remote_memory = make_remote_memory(config.rl_config)

    # 初期パラメータ
    if init_parameter is not None:
        logger.info("init parameter set")
        board.write(init_parameter.backup())
    if init_remote_memory is not None:
        remote_memory.restore(init_remote_memory.backup())
        logger.info(f"init remote_memory set(len={remote_memory.length()})")

    MPManager.register("get_config", callable=lambda: config)
    MPManager.register("get_options", callable=lambda: options)
    MPManager.register("ServerManager", callable=lambda: server_manager)
    MPManager.register("RemoteMemory", callable=lambda: remote_memory)
    MPManager.register("Board", callable=lambda: board)

    manager = MPManager(address=(config.remote_server_ip, config.remote_port), authkey=config.remote_authkey)
    server: Any = manager.get_server()

    # add server stop function
    shutdown_timer = threading.Timer(1, lambda: server.stop_event.set())
    MPManager.register("server_stop", callable=lambda: shutdown_timer.start())

    logger.info(f"--- server start({config.remote_server_ip}:{config.remote_port}) ---")
    server.serve_forever()


# ----------------------------
# 学習
# ----------------------------

__is_set_start_method = False


def train(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
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
    queue_timeout: int = 60 * 10,
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    global __is_set_start_method

    assert timeout != -1 or max_train_count != -1, "Please specify 'timeout' or 'max_train_count'."

    config.init_play()
    config = config.copy(env_share=False)

    # stop config
    config._max_episodes = max_episodes
    config._timeout = timeout
    config._max_steps = max_steps
    config._max_train_count = max_train_count
    # play config
    config._shuffle_player = shuffle_player
    config._disable_trainer = False
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

    if not __is_set_start_method:
        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn", force=True)
            __is_set_start_method = True

    # callbacks
    _info = {
        "config": config,
    }
    [c.on_init(_info) for c in config.callbacks]

    # --- last params (errorチェックのため先に作っておく)
    last_parameter = config.make_parameter()
    last_remote_memory = config.make_remote_memory()

    # --- create process
    last_param_queue = mp.Queue()
    last_remote_memory_queue = mp.Queue()
    ps_trainer = mp.Process(
        target=__run_trainer,
        args=(
            last_param_queue,
            last_remote_memory_queue,
            config,
            options,
            save_remote_memory,
        ),
    )
    ps_server = mp.Process(
        target=__run_server,
        args=(
            config,
            options,
            parameter,
            remote_memory,
        ),
    )

    # -------------------------------------
    # wait server
    # -------------------------------------
    ps_server.start()
    time.sleep(1)

    MPManager.register("ServerManager")
    MPManager.register("server_stop")
    manager: Any = MPManager(address=("127.0.0.1", config.remote_port), authkey=config.remote_authkey)
    manager.connect()
    server_manager: ServerManager = manager.ServerManager()

    # --- wait loop
    print(f"Wait ActorPC: {config.actor_num}")
    print("  ActorPC should be connected to this PC.")
    print("  See 'examples/sample_remote_actor.py' for connection instructions.")
    while True:
        time.sleep(1)
        if server_manager.get_actors_count() >= config.actor_num:
            print("Start Train.")
            server_manager.start()
            break

    # -------------------------------------
    # train
    # -------------------------------------
    ps_trainer.start()

    # callbacks
    [c.on_start(_info) for c in config.callbacks]

    # --- wait loop
    t0 = time.time()
    while True:
        time.sleep(1)  # polling time

        if not ps_trainer.is_alive():
            server_manager.set_end()
            logger.info("train end(trainer process dead)")
            break

        if not ps_server.is_alive():
            logger.info("train end(server process dead)")
            break

        if server_manager.get_actors_count() == 0:
            server_manager.set_end()
            logger.info("train end(actors dead)")
            break

        # callbacks
        [c.on_polling(_info) for c in config.callbacks]

        if server_manager.is_end():
            break
    logger.info(f"wait loop end.(run time: {(time.time() - t0)/60:.2f}m)")

    # --- Trainerから最終parameterを取得する
    try:
        t0 = time.time()
        params = last_param_queue.get(timeout=queue_timeout)
        if params is not None:
            last_parameter.restore(params)
        logger.info(f"recv parameter time: {time.time() - t0:.1f}s")
    except Exception:
        logger.warning(traceback.format_exc())

    # --- last memory
    try:
        if return_remote_memory:
            t0 = time.time()
            dat = last_remote_memory_queue.get(timeout=queue_timeout)
            if dat is not None:
                last_remote_memory.restore(dat)
            logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s")
    except Exception:
        logger.warning(traceback.format_exc())

    # --- server プロセスを終了させる
    manager.server_stop()

    # --- プロセスの終了を待つ
    for _ in range(60 * 10):
        if ps_server.is_alive():
            time.sleep(1)
        else:
            break
    else:
        ps_server.terminate()
    for _ in range(60 * 10):
        if ps_trainer.is_alive():
            time.sleep(1)
        else:
            break
    else:
        ps_trainer.terminate()

    # 子プロセスが正常終了していなければ例外を出す
    # exitcode: 0 正常, 1 例外, 負 シグナル
    if ps_server is not None and ps_server.exitcode != 0:
        raise RuntimeError(f"An exception has occurred in server process.(exitcode: {ps_server.exitcode})")
    if ps_trainer is not None and ps_trainer.exitcode != 0:
        raise RuntimeError(f"An exception has occurred in trainer process.(exitcode: {ps_trainer.exitcode})")

    # --- callbacks
    try:
        [c.on_end(_info) for c in config.callbacks]
    except Exception:
        logger.warning(traceback.format_exc())

    # --- history
    return_history = HistoryViewer()
    try:
        if history is not None:
            return_history.load(config.save_dir)
    except Exception:
        logger.info(traceback.format_exc())

    return last_parameter, last_remote_memory, return_history
