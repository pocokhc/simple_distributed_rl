import ctypes
import multiprocessing as mp
import unittest
from multiprocessing.managers import BaseManager

import gym
from srl import rl
from srl.base.rl.env_for_rl import EnvForRL
from srl.base.rl.rl import RLRemoteMemory


def _run_episode(
    env,
    worker,
    rendering=False,
):

    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    valid_actions = env.fetch_valid_actions()
    worker.on_reset(state, valid_actions, env)

    for step in range(env.max_episode_steps):

        # action
        env_action, worker_action = worker.policy(state, valid_actions, env)
        if valid_actions is not None:
            assert env_action in valid_actions

        # env step
        next_state, reward, done, env_info = env.step(env_action)
        step += 1
        total_reward += reward
        next_valid_actions = env.fetch_valid_actions()

        # rl step
        work_info = worker.on_step(
            state, worker_action, next_state, reward, done, valid_actions, next_valid_actions, env
        )

        # render
        if rendering:
            env.render()
            worker.render(state, valid_actions, env.action_to_str)
            print(
                "{} action {}, reward: {}, done: {}, info: {} {}".format(
                    step, env_action, reward, done, env_info, work_info
                )
            )

        # step after
        if done:
            break
        state = next_state
        valid_actions = next_valid_actions

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

    env = EnvForRL(gym.make(env_name), rl_config)
    rl_module = rl.make(rl_config.getName())
    parameter = rl_module.Parameter(rl_config)
    worker = rl_module.Worker(rl_config, parameter, remote_memory, worker_id)
    worker.set_training(True)

    prev_update_count = 0
    episode = 0

    # episode loop
    while True:
        if train_end_signal.value:
            break

        step, reward = _run_episode(env, worker)
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
    last_param_q: mp.Queue,
    train_end_signal: ctypes.c_bool,
):
    rl_config = config["rl_config"]

    rl_module = rl.make(rl_config.getName())
    parameter = rl_module.Parameter(rl_config)
    trainer = rl_module.Trainer(rl_config, parameter, remote_memory)

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
    last_param_q.put(parameter.backup())


class MPManager(BaseManager):
    pass


def main():

    # --- config
    env_name = "FrozenLake-v1"
    rl_config = rl.ql.Config()
    worker_num = 2
    config = {
        "env_name": env_name,
        "rl_config": rl_config,
        "max_train_count": 100000,
        "trainer_parameter_send_interval_by_train_count": 100,
    }

    # init
    rl_config.assert_params()
    rl_module = rl.make(rl_config.getName())
    env = EnvForRL(gym.make(env_name), rl_config)  # (rl_config init by env)

    # --- async
    MPManager.register("RemoteMemory", rl_module.RemoteMemory)
    MPManager.register("Board", Board)

    with MPManager() as manager:
        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        remote_memory = manager.RemoteMemory(rl_config)
        remote_board = manager.Board()
        last_param_q = mp.Queue()

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
            last_param_q,
            train_end_signal,
        )
        trainer_ps = mp.Process(target=_run_trainer, args=params)

        # --- start
        [p.start() for p in workers_ps_list]
        trainer_ps.start()
        trainer_ps.join()

        # 学習後の結果
        param = last_param_q.get()
        last_param_q.close()
        parameter = rl_module.Parameter(rl_config)
        parameter.restore(param)

        # 強制終了
        [p.terminate() for p in workers_ps_list]
        trainer_ps.terminate()

    # --- rendering
    worker = rl_module.Worker(rl_config, parameter, None, 0)
    worker.set_training(False)
    step, reward = _run_episode(env, worker, rendering=True)
    print(f"step: {step}, reward: {reward}")


class Test(unittest.TestCase):
    def test_run(self):
        main()


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_run", verbosity=2)
