import logging
import multiprocessing as mp
import os
import pprint
import socket
import threading
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, Optional, cast

import numpy as np

from srl.base.rl.base import RLParameter
from srl.base.rl.registration import make_remote_memory
from srl.runner.callback import Callback, MPCallback
from srl.runner.runner import Config, Context, Runner

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
class _ActorInterrupt(Callback):
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

    def on_episodes_begin(self, runner: Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: Runner):
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
        runner.state.sync_actor += 1

    def intermediate_stop(self, runner: Runner) -> bool:
        return self.server_manager.is_end()


def run_actor(
    server_ip: str,
    port: int,
    authkey: bytes = b"abracadabra",
    actor_id: Optional[int] = None,
    verbose: bool = True,
):
    MPManager.register("get_config")
    MPManager.register("get_context")
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
    context: Context = manager.get_context().copy()
    server_manager: ServerManager = manager.ServerManager()
    remote_memory = manager.RemoteMemory()
    remote_board: Board = manager.Board()

    logger.info(f"Config\n{pprint.pformat(config.to_dict())}")
    logger.info(f"Context\n{pprint.pformat(context.to_dict())}")

    runner = Runner(config.env_config, config.rl_config)
    runner.config = config
    runner.context = context

    # --- env check
    try:
        runner.make_env()
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
        context.run_name = f"actor{actor_id}"
        context.actor_id = actor_id

        s = f"Joined server(Your actor index is {actor_id})"
        logger.info(s)
        if verbose:
            print(s)

        # --- set_config_by_actor
        config.rl_config.set_config_by_actor(config.actor_num, actor_id)

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        runner.context.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                server_manager,
                config,
            )
        )

        # --- wait
        print("Waiting for server to start...")
        while True:
            if server_manager.is_running():
                break
            time.sleep(1)  # polling

        # --- play
        context.train_only = False
        context.disable_trainer = True
        context.training = True
        runner._play(parameter, remote_memory)

    finally:
        server_manager.disconnected(actor_id)
        logger.info(f"actor{context.actor_id} end")


# --------------------
# trainer
# --------------------
class _TrainerInterrupt(Callback):
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
        self.trainer_parameter_send_interval_by_train_count = config.trainer_parameter_send_interval_by_train_count

    def on_trainer_start(self, runner: Runner):
        runner.state.sync_trainer = 0

    def on_trainer_train(self, runner: Runner):
        train_count = runner.state.trainer.get_train_count()  # type:ignore , trainer is not None

        if train_count == 0:
            time.sleep(1)
            return

        if train_count % self.trainer_parameter_send_interval_by_train_count == 0:
            self.remote_board.write(self.parameter.backup())
            runner.state.sync_trainer += 1

    def intermediate_stop(self, info) -> bool:
        return self.server_manager.is_end()


def __run_trainer(
    config: Config,
    context: Context,
    last_param_queue: mp.Queue,
    last_remote_memory_queue: mp.Queue,
):
    context.run_name = "trainer"
    runner = Runner(config.env_config, config.rl_config)
    runner.config = config
    runner.context = context
    logger.info("trainer start.")

    MPManager.register("ServerManager")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")

    manager: Any = MPManager(address=("127.0.0.1", config.remote_port), authkey=config.remote_authkey)
    manager.connect()
    server_manager: ServerManager = manager.ServerManager()

    parameter = None
    remote_memory = None
    try:
        remote_memory = manager.RemoteMemory()
        remote_board: Board = manager.Board()

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        runner.context.callbacks.append(
            _TrainerInterrupt(
                remote_board,
                parameter,
                server_manager,
                config,
            )
        )

        # --- play
        context.train_only = True
        context.disable_trainer = False
        context.training = True
        runner._play(parameter, remote_memory)

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
            # if remote_memory is not None and save_remote_memory != "":
            #    remote_memory.save(save_remote_memory, compress=True)

            # --- send last memory
            if remote_memory is not None:
                t0 = time.time()
                last_remote_memory_queue.put(remote_memory.backup())
                logger.info(f"send remote_memory time: {time.time() - t0:.1f}s)")


# --------------------
# server
# --------------------
def __run_server(config: Config, context: Context):
    # とりあえずCPUにしておく
    if config.use_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("[server] set CUDA_VISIBLE_DEVICES=-1")

    server_manager = ServerManager()
    board = Board()
    remote_memory = make_remote_memory(config.rl_config, is_load=True)

    MPManager.register("get_config", callable=lambda: config)
    MPManager.register("get_context", callable=lambda: context)
    MPManager.register("ServerManager", callable=lambda: server_manager)
    MPManager.register("RemoteMemory", callable=lambda: remote_memory)
    MPManager.register("Board", callable=lambda: board)

    manager = MPManager(address=("", config.remote_port), authkey=config.remote_authkey)
    server: Any = manager.get_server()

    # add server stop function
    shutdown_timer = threading.Timer(1, lambda: server.stop_event.set())
    MPManager.register("server_stop", callable=lambda: shutdown_timer.start())

    ip = socket.gethostbyname(socket.gethostname())
    logger.info(f"--- server start({ip}:{config.remote_port}) ---")
    server.serve_forever()


# ----------------------------
# 学習
# ----------------------------

__is_set_start_method = False


def train(
    runner: Runner,
    queue_timeout: int = 60 * 10,
):
    global __is_set_start_method

    config = runner.config
    context = runner.context

    logger.info(f"Config\n{pprint.pformat(config.to_dict())}")
    logger.info(f"Context\n{pprint.pformat(context.to_dict())}")

    if not __is_set_start_method:
        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn", force=True)
            __is_set_start_method = True

    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- last
    last_parameter = runner.make_parameter()
    last_remote_memory = runner.make_remote_memory()

    # --- create process
    last_param_queue = mp.Queue()
    last_remote_memory_queue = mp.Queue()
    ps_trainer = mp.Process(
        target=__run_trainer,
        args=(
            config,
            context,
            last_param_queue,
            last_remote_memory_queue,
        ),
    )
    ps_server = mp.Process(
        target=__run_server,
        args=(
            config,
            context,
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
    [c.on_start(runner) for c in _callbacks]

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
        [c.on_polling(runner) for c in _callbacks]

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
    if False:  # TODO
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

    # callbacks
    [c.on_end(runner) for c in _callbacks]
