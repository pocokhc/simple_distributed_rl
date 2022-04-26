import ctypes
import multiprocessing as mp
import os
from multiprocessing.managers import BaseManager

import srl
from srl.base.env.env_for_rl import EnvForRL
from srl.base.env.singleplay_wrapper import SinglePlayerWrapper
from srl.base.rl.base import RLRemoteMemory
from srl.base.rl.registration import make_parameter, make_remote_memory, make_trainer, make_worker
from srl.rl.processor import ContinuousProcessor, DiscreteProcessor, ObservationBoxProcessor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _run_episode(
    env,
    worker,
    trainer,
    training,
    rendering=False,
):
    worker.set_training(training)
    env = SinglePlayerWrapper(env)  # change single play interface

    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    invalid_actions = env.fetch_invalid_actions()
    worker.on_reset(state, invalid_actions, env, [0])

    if rendering:
        print("step 0")
        env.render()

    while True:

        # render
        if rendering:
            worker.render(state, invalid_actions, env)

        # action
        env_action, worker_action = worker.policy(state, invalid_actions, env, [0])
        assert env_action not in invalid_actions

        # env step
        next_state, reward, done, env_info = env.step(env_action)
        step += 1
        total_reward += reward
        next_invalid_actions = env.fetch_invalid_actions()

        if step > env.max_episode_steps:
            done = True

        # rl step
        work_info = worker.on_step(
            state, worker_action, next_state, reward, done, invalid_actions, next_invalid_actions, env
        )

        # train
        if training and trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # render
        if rendering:
            print(
                "step {}, action {}, reward: {}, done: {}, info: {} {} {}".format(
                    step, env_action, reward, done, env_info, work_info, train_info
                )
            )
            env.render()

        # step after
        if done:
            break
        state = next_state
        invalid_actions = next_invalid_actions

    return step, total_reward


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


def _run_worker(
    config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    worker_id: int,
    train_end_signal: ctypes.c_bool,
):
    env_name = config["env_name"]
    rl_config = config["rl_config"]

    env = srl.envs.make(env_name)
    env = EnvForRL(env, rl_config, processors=config["processors"])

    parameter = make_parameter(rl_config, None)
    worker = make_worker(rl_config, None, parameter, remote_memory, worker_id)

    prev_update_count = 0
    episode = 0

    # episode loop
    while True:
        if train_end_signal.value:
            break

        step, reward = _run_episode(env, worker, trainer=None, training=True)
        episode += 1

        # sync parameter
        update_count = remote_board.get_update_count()
        if update_count != prev_update_count:
            prev_update_count = update_count
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)

        if episode % 1000 == 0:
            print(f"{worker_id}: {episode} episode, {step} step, {reward} reward")


def _run_trainer(
    config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    rl_config = config["rl_config"]

    parameter = make_parameter(rl_config, None)
    trainer = make_trainer(rl_config, None, parameter, remote_memory)

    train_count = 0
    while True:
        if train_end_signal.value:
            break

        if train_count >= config["max_train_count"]:
            break

        trainer.train()
        train_count = trainer.get_train_count()

        # send parameter
        if train_count % config["trainer_parameter_send_interval_by_train_count"] == 0:
            remote_board.write(parameter.backup())

        if train_count > 0 and train_count % 10000 == 0:
            print(f"{train_count} / 100000 train")

    train_end_signal.value = True

    # 学習結果を送信
    remote_board.write(parameter.backup())


class MPManager(BaseManager):
    pass


def main():

    # --- config
    env_name = "FrozenLake-v1"
    rl_config = srl.rl.ql.Config()
    worker_num = 2
    config = {
        "env_name": env_name,
        "rl_config": rl_config,
        "max_train_count": 100000,
        "trainer_parameter_send_interval_by_train_count": 100,
        "processors": [
            ObservationBoxProcessor(),
            DiscreteProcessor(),
            ContinuousProcessor(),
        ],
    }

    # config init
    rl_config.assert_params()
    env = srl.envs.make(env_name)
    env = EnvForRL(env, rl_config, processors=config["processors"])

    # --- async
    MPManager.register("RemoteMemory", make_remote_memory(rl_config, env, get_class=True))
    MPManager.register("Board", Board)

    with MPManager() as manager:
        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        remote_memory = manager.RemoteMemory(rl_config)
        remote_board = manager.Board()

        # --- worker
        workers_ps_list = []
        for worker_id in range(worker_num):
            params = (
                config,
                remote_memory,
                remote_board,
                worker_id,
                train_end_signal,
            )
            ps = mp.Process(target=_run_worker, args=params)
            workers_ps_list.append(ps)

        # --- trainer
        params = (
            config,
            remote_memory,
            remote_board,
            train_end_signal,
        )
        trainer_ps = mp.Process(target=_run_trainer, args=params)

        # --- start
        [p.start() for p in workers_ps_list]
        trainer_ps.start()
        trainer_ps.join()

        # 学習後の結果
        parameter = make_parameter(rl_config, env)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # 強制終了
        [p.terminate() for p in workers_ps_list]
        trainer_ps.terminate()

    # --- rendering
    worker = make_worker(rl_config, env, parameter)
    step, reward = _run_episode(env, worker, trainer=None, training=False, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
