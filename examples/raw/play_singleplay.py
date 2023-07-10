from typing import Optional

import numpy as np

import srl
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    worker: WorkerRun,
    trainer: Optional[RLTrainer],
):
    # 1. reset
    env.reset()
    worker.on_reset(player_index=0, training=(trainer is not None))

    while not env.done:
        # 2. action
        action = worker.policy()

        # 3. step
        env.step(action)
        worker.on_step()

        # 4. train
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # --- step info
        if False:
            print(env.state)
            print(env.reward)
            print(env.done)
            print(env.info)
            print(worker.info)
            print(train_info)

    return env.step_num, env.episode_rewards[0]


def _render(env: EnvRun, worker: WorkerRun):
    # 1. reset
    # (Do not set both render_mode to window mode)
    env.reset(render_mode="terminal")
    worker.on_reset(player_index=0, training=False, render_mode="terminal")

    # --- render
    print("step 0")
    env.render()

    while not env.done:
        # 2. action
        action = worker.policy()

        # --- worker render
        worker.render()

        # 3. step
        env.step(action)
        worker.on_step()

        # --- render
        print(f"step        :{env.step_num}")
        print(f"action      :{action}")
        print(f"reward      :{env.reward} {env.step_rewards}")
        print(f"env info    :{env.info}")
        print(f"worker info :{worker.info}")

        env.render()

    print(f"done reason : {env.done_reason}")
    print(f"step        : {env.step_num}")
    print(f"reward      : {env.episode_rewards[0]}")


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # env init
    env = srl.make_env(env_config)
    rl_config.reset(env)

    # make instance
    parameter = srl.make_parameter(rl_config)
    remote_memory = srl.make_remote_memory(rl_config)
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)
    worker = srl.make_worker(rl_config, env, parameter, remote_memory)

    # --- train loop
    for episode in range(10000):
        step, reward = _run_episode(env, worker, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward:.5f} reward")

    # --- evaluate
    reward_list = []
    for episode in range(100):
        _, reward = _run_episode(env, worker, None)
        reward_list.append(reward)
    print(f"Average reward for 100 episodes: {np.mean(reward_list):.5f}")

    # --- render
    _render(env, worker)


if __name__ == "__main__":
    common.logger_print()

    main()
