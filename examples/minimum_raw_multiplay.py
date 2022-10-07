from typing import List

import numpy as np
import srl
import srl.rl.random_play
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.worker import WorkerRun

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _train(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
):
    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter, remote_memory, training=True, distributed=False),
        srl.make_worker_rulebase("random"),
    ]
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)

    # 1. reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy(env)

        # 3. step
        env.step(action)
        [w.on_step(env) for w in workers]

        # 4. train
        train_info = trainer.train()

    return env.step_num, env.episode_rewards


def _run_episode(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
):
    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter, remote_memory=None, training=False, distributed=False),
        srl.make_worker_rulebase("random"),
    ]

    # 1. reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy(env)

        # 3. step
        env.step(action)
        [w.on_step(env) for w in workers]

    return env.episode_rewards


def _render(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
):
    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter, remote_memory=None, training=False, distributed=False),
        srl.make_worker_rulebase("random"),
    ]

    # --- set render mode
    env.set_render_mode("terminal")
    [w.set_render_mode("terminal") for w in workers]

    # 1. reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    # --- render
    print("step 0")
    env.render()

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy(env)

        # --- worker render
        print(f"player {env.next_player_index}")
        workers[env.next_player_index].render(env)

        # 3. step
        env.step(action)
        [w.on_step(env) for w in workers]

        # --- env render
        print(
            "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                env.step_num, action, env.step_rewards, env.done, env.next_player_index, env.info
            )
        )
        print("player {} info: {}".format(env.next_player_index, workers[env.next_player_index].info))
        env.render()


def main():

    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # env init
    env = srl.make_env(env_config)

    # rl init
    rl_config.reset_config(env)
    parameter = srl.make_parameter(rl_config)
    remote_memory = srl.make_remote_memory(rl_config)

    # --- train loop
    for episode in range(10000):
        step, reward = _train(env, rl_config, parameter, remote_memory)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- evaluate
    rewards_list = []
    for episode in range(100):
        rewards = _run_episode(env, rl_config, parameter)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")

    # --- render
    _render(env, rl_config, parameter)


if __name__ == "__main__":
    main()
