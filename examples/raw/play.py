from typing import Optional

import numpy as np

import srl
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    worker: WorkerRun,
    trainer: Optional[RLTrainer],
    rendering: bool,
):
    # 1. reset
    env.reset()
    worker.reset(player_index=0)

    action = None
    while not env.done:
        # 2. action
        action = worker.policy()

        # --- env render
        if rendering:
            print(f"\n--- turn {env.step_num}, action {action}, rewards: {env.rewards[0]}, done: {env.done}, info: {env.info}")
            print(f"worker: {worker.info}")
            env.render()
            worker.render()

        # 3. step
        env.step(action)
        worker.on_step()

        # 4. train
        if trainer is not None:
            trainer.train()

        # --- step info
        if False:
            print(env.state)
            print(env.reward)
            print(env.done)
            print(env.info)
            print(worker.info)
            if trainer is not None:
                print(trainer.info)

    # --- last env render
    if rendering:
        print(f"\n--- turn: {env.step_num}, reward: {env.rewards[0]}, total reward: {env.episode_rewards[0]}, done reason: {env.done_reason}")
        env.render()

    return env.step_num, env.episode_rewards


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # --- make instance
    env = env_config.make()
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()
    trainer = rl_config.make_trainer(parameter, memory)
    worker = rl_config.make_worker(env, parameter, memory)

    # --- train
    context = srl.RunContext(env_config, rl_config, training=True)
    env.setup(context)
    worker.setup(context)
    trainer.setup(context)
    for episode in range(10000):
        step, rewards = _run_episode(env, worker, trainer, rendering=False)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")
    env.teardown()
    worker.teardown()
    trainer.teardown()

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    env.setup(context)
    worker.setup(context)
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, worker, None, rendering=False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")
    env.teardown()
    worker.teardown()

    # --- render
    context = srl.RunContext(env_config, rl_config, env_render_mode="terminal", rl_render_mode="terminal")
    env.setup(context)
    worker.setup(context)
    _run_episode(env, worker, None, rendering=True)
    env.teardown()
    worker.teardown()


if __name__ == "__main__":
    common.logger_print()

    main()
