import numpy as np

import srl
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLConfig, RLParameter

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    training: bool,
):
    # 0. make
    remote_memory = None
    trainer = None
    if training:
        remote_memory = srl.make_remote_memory(rl_config)
        trainer = srl.make_trainer(rl_config, parameter, remote_memory)

    worker = srl.make_worker(rl_config, parameter, remote_memory, training=training, distributed=False)

    # 1. reset
    env.reset()
    worker.on_reset(env)

    while not env.done:
        # 2. action
        action = worker.policy(env)

        # 3. step
        env.step(action)
        worker.on_step(env)

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


def _render(env: EnvRun, rl_config: RLConfig, parameter: RLParameter):
    worker = srl.make_worker(rl_config, parameter)

    # 1. reset
    env.reset(render_mode="terminal")
    worker.on_reset(env, render_mode="terminal")

    # --- render
    print("step 0")
    env.render()

    while not env.done:
        # 2. action
        action = worker.policy(env)

        # --- worker render
        worker.render(env)

        # 3. step
        env.step(action)
        worker.on_step(env)

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

    # make parameter
    parameter = srl.make_parameter(rl_config)

    # --- train loop
    for episode in range(10000):
        step, reward = _run_episode(env, rl_config, parameter, True)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward:.5f} reward")

    # --- evaluate
    reward_list = []
    for episode in range(100):
        _, reward = _run_episode(env, rl_config, parameter, False)
        reward_list.append(reward)
    print(f"Average reward for 100 episodes: {np.mean(reward_list):.5f}")

    # --- render
    _render(env, rl_config, parameter)


if __name__ == "__main__":
    main()
