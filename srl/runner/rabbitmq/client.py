import logging
import time
import traceback
from typing import List, Tuple

import srl
from srl.runner.rabbitmq.rabbitmq_manager import RabbitMQManager

logger = logging.getLogger(__name__)


def _assign_member(
    mq: RabbitMQManager,
    users: List[Tuple[str, str]],
    active_uids: List[str],
    check_id: str,
    target_role: str,
):
    is_update_board = False

    # alive check
    if check_id != "":
        if check_id not in active_uids:
            check_id = ""
            is_update_board = True
        else:
            # 相手のboardがRUNで自分じゃないならアサイン取り消し
            _b = mq.board_get(check_id)
            if _b is not None and _b["status"] == "RUN":
                if _b["client"] != mq.uid:
                    check_id = ""
                    is_update_board = True

    # assign
    if check_id == "":
        for role, uid in users:
            if role != target_role:
                continue
            # 相手がokな状態か
            _b = mq.board_get(uid)
            if _b is not None and _b["status"] == "":
                check_id = uid
                is_update_board = True

    return check_id, is_update_board


def run(
    runner: srl.Runner,
    host: str,
    port: int = 5672,
    username: str = "guest",
    password: str = "guest",
    virtual_host: str = "/",
):
    parameter = runner.make_parameter()

    mq = RabbitMQManager(host, port, username, password, virtual_host)
    mq.join("client")

    check_interval = 10  # TODO

    mq.create_fanout_queue_once_if_not_exists(f"parameter_{mq.uid}", "parameter")
    mq.create_queue_once_if_not_exists(f"last_parameter_{mq.uid}")
    mq.create_queue_once_if_not_exists(f"memory_{mq.uid}")

    # --- board
    board = {
        "status": "",
        "actor_ids": ["" for _ in range(runner.context.actor_num)],
        "trainer_id": "",
        "mp_data": runner.create_mp_data(),
    }
    assert mq.board_update(board)
    is_update_board = False

    # --- run
    _check_t0 = time.time()
    try:
        print("wait main")
        logger.info("wait main")
        while True:
            time.sleep(1)
            if mq.keepalive():
                mq.health_check()

            # --- member assign ---
            users = mq.fetch_users()
            active_uids = [uid for role, uid in users]
            _id, _f = _assign_member(mq, users, active_uids, board["trainer_id"], "trainer")
            if _f:
                board["trainer_id"] = _id
                is_update_board = True
                logger.info(f"trainer assign: {_id}")
            for i in range(runner.context.actor_num):
                _id, _f = _assign_member(mq, users, active_uids, board["actor_ids"][i], "actor")
                if _f:
                    board["actor_ids"][i] = _id
                    is_update_board = True
                    logger.info(f"actor{i} assign: {_id}")

            if is_update_board:
                if mq.board_update(board):
                    is_update_board = False
            # -------------------------

            # --- end check
            mq.create_queue_once_if_not_exists(f"last_parameter_{mq.uid}")
            params = mq.recv_once_lastdata_and_purge(f"last_parameter_{mq.uid}")
            if params is not None:
                try:
                    parameter.restore(params)
                    print("main end")
                    logger.info("main end")
                except Exception:
                    logger.warning(traceback.format_exc())
                finally:
                    break

            # --- progress
            if time.time() - _check_t0 > check_interval:
                try:
                    _check_t0 = time.time()

                    print("myid", mq.uid)
                    print("status", board["status"])
                    print("actor_ids", board["actor_ids"])
                    print("trainer_id", board["trainer_id"])

                    users = mq.fetch_users()
                    for role, uid in users:
                        if role == "client":
                            continue
                        if uid == mq.uid:
                            continue
                        _b = mq.board_get(uid)
                        print(role, uid, _b)

                    # --- check param
                    # 定期的にparameterを取得
                    # mq.create_fanout_queue_once_if_not_exists(f"parameter_{mq.uid}", "parameter")
                    # params = mq.recv_once_lastdata_and_purge(f"parameter_{mq.uid}")
                    # if params is not None:
                    #    parameter.restore(params)
                    #    print(runner.evaluate())

                    qsize = mq.fetch_qsize_once(f"memory_{mq.uid}")
                    if qsize is not None:
                        print(f"memory: {qsize}")

                except Exception:
                    logger.warning(traceback.format_exc())

    finally:
        mq.leave()
