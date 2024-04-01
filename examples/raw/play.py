from typing import List, Optional

import numpy as np

import srl
import srl.rl.random_play
from srl.base.context import RunContext
from srl.base.env.env_run import EnvRun
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    workers: List[WorkerRun],
    trainer: Optional[RLTrainer],
    rendering: bool,
):
    # 1. reset
    env.reset()
    [w.on_reset(player_index=i) for i, w in enumerate(workers)]

    # --- env render
    if rendering:
        print("step 0")
        env.render()

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy()

        # --- worker render
        if rendering:
            print(f"player {env.next_player_index}")
            workers[env.next_player_index].render()

        # 3. step
        env.step(action)
        [w.on_step() for w in workers]

        # --- env render
        if rendering:
            print(
                "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, action, env.step_rewards, env.done, env.next_player_index, env.info
                )
            )
            print("player {} info: {}".format(env.next_player_index, workers[env.next_player_index].info))
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
            print(workers[0].info)
            print(workers[1].info)
            print(train_info)

    if rendering:
        print(f"step: {env.step_num}, reward: {env.episode_rewards}")

    return env.step_num, env.episode_rewards


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # --- make instance
    env = srl.make_env(env_config)
    rl_config.setup(env)
    parameter = srl.make_parameter(rl_config)
    memory = srl.make_memory(rl_config)
    trainer = srl.make_trainer(rl_config, parameter, memory)
    workers = [
        srl.make_worker(rl_config, env, parameter, memory),
        srl.make_worker_rulebase("random", env),
    ]

    # --- train
    # set context
    context = RunContext(training=True)
    env.setup(context)
    [w.on_start(context) for w in workers]
    trainer.train_start(context)
    # loop
    for episode in range(10000):
        step, rewards = _run_episode(env, workers, trainer, rendering=False)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")

    # --- evaluate
    # set context
    context = RunContext()
    env.setup(context)
    [w.on_start(context) for w in workers]
    trainer.train_start(context)
    # run
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, workers, None, rendering=False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")

    # --- render
    # set context
    context = RunContext(render_mode="terminal")
    env.setup(context)
    [w.on_start(context) for w in workers]
    trainer.train_start(context)
    # run
    _run_episode(env, workers, None, rendering=True)


def play_cpu():
    import srl.rl.human

    env_config = srl.EnvConfig("OX")

    # make
    env = srl.make_env(env_config)
    player = srl.make_worker_rulebase("human", env)
    cpu = env.make_worker("cpu")
    assert cpu is not None

    # set context
    context = RunContext(render_mode="terminal")
    env.setup(context)
    player.on_start(context)
    cpu.on_start(context)

    # reset
    env.reset()
    player.on_reset(player_index=0)
    cpu.on_reset(player_index=1)

    env.render()

    while not env.done:
        # 1 step
        if env.next_player_index == 0:
            action = player.policy()
        else:
            action = cpu.policy()
        env.step(action)
        player.on_step()
        cpu.on_step()

        env.render()


if __name__ == "__main__":
    common.logger_print()

    main()
    play_cpu()
