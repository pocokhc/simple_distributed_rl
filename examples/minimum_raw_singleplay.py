from typing import Optional

import srl
from srl.base.env.base import EnvRun
from srl.base.env.singleplay_wrapper import SinglePlayEnvWrapper
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.singleplay_wrapper import SinglePlayWorkerWrapper

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _train(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
):
    worker = srl.make_worker(rl_config, parameter, remote_memory, training=True, distributed=False)
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)

    # --- change single play interface
    env = SinglePlayEnvWrapper(env)
    worker = SinglePlayWorkerWrapper(worker)

    # 1. reset
    state = env.reset()
    worker.on_reset(env)

    while not env.done:
        # 2. action
        action = worker.policy(env)

        # 3. step
        state, reward, done, env_info = env.step(action)
        work_info = worker.on_step(env)

        # 4. train
        train_info = trainer.train()

    return env.step_num, env.episode_rewards[0]


def _render(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
):
    worker = srl.make_worker(rl_config, parameter, remote_memory=None, training=False, distributed=False)

    # --- change single play interface
    env = SinglePlayEnvWrapper(env)
    worker = SinglePlayWorkerWrapper(worker)

    # 1. reset
    state = env.reset(mode="terminal")
    worker.on_reset(env, mode="terminal")

    # --- render
    print("step 0")
    env.render()

    while not env.done:
        # 2. action
        action = worker.policy(env)

        # --- worker render
        worker.render(env)

        # 3. step
        state, reward, done, env_info = env.step(action)
        work_info = worker.on_step(env)

        # --- env render
        print(
            "step {}, action {}, reward: {}, done: {}, info: {} {}".format(
                env.step_num, action, env.step_rewards[0], env.done, env_info, work_info
            )
        )
        env.render()

    print(f"step: {env.step_num}, reward: {env.episode_rewards[0]}")


def main():

    env_config = srl.EnvConfig("Grid")
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

    # --- render
    _render(env, rl_config, parameter)


if __name__ == "__main__":
    main()
