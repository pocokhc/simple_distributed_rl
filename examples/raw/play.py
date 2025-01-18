from typing import Optional

import numpy as np

import srl
import srl.rl.random_play
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
    worker.on_reset(player_index=0)

    # --- env render
    if rendering:
        print("step 0")
        env.render()

    while not env.done:
        # 2. action
        action = worker.policy()

        # --- worker render
        if rendering:
            print(f"player {env.next_player}")
            worker.render()

        # 3. step
        env.step(action)
        worker.on_step()

        # --- env render
        if rendering:
            print("--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(env.step_num, action, env.rewards, env.done, env.next_player, env.info))
            print("player {} info: {}".format(env.next_player, worker.info))
            env.render()

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

    if rendering:
        print(f"step: {env.step_num}, reward: {env.episode_rewards}")

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
    worker.on_start(context)
    trainer.on_start(context)
    for episode in range(10000):
        step, rewards = _run_episode(env, worker, trainer, rendering=False)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")
    worker.on_end()
    trainer.on_end()

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    env.setup(context)
    worker.on_start(context)
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, worker, None, rendering=False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")
    worker.on_end()

    # --- render
    context = srl.RunContext(env_config, rl_config, render_mode="terminal")
    env.setup(context)
    worker.on_start(context)
    _run_episode(env, worker, None, rendering=True)
    worker.on_end()


if __name__ == "__main__":
    common.logger_print()

    main()
