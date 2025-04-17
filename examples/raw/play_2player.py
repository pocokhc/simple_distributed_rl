from typing import List, Optional

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
    workers: List[WorkerRun],
    trainer: Optional[RLTrainer],
    rendering: bool,
):
    # 1. reset
    env.reset()
    [w.reset(player_index=i) for i, w in enumerate(workers)]

    action = None
    while not env.done:
        # 2. action
        action = workers[env.next_player].policy()

        # --- render
        if rendering:
            print("\n--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(env.step_num, action, env.rewards, env.done, env.next_player, env.info))
            print("player {} info: {}".format(env.next_player, workers[env.next_player].info))
            env.render()
            workers[env.next_player].render()

        # 3. step
        env.step(action)
        [w.on_step() for w in workers]

        # 4. train
        if trainer is not None:
            trainer.train()

        # --- step info
        if False:
            print(env.state)
            print(env.reward)
            print(env.done)
            print(env.info)
            print(workers[0].info)
            print(workers[1].info)
            if trainer is not None:
                print(trainer.info)

    if rendering:
        print(f"\n--- turn: {env.step_num}, reward: {env.rewards}, total reward: {env.episode_rewards}, done reason: {env.done_reason}")
        env.render()

    return env.step_num, env.episode_rewards


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # --- make instance
    env = env_config.make()
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()
    trainer = rl_config.make_trainer(parameter, memory)
    workers = [
        rl_config.make_worker(env, parameter, memory),
        srl.make_worker("random", env),
    ]

    # --- train
    context = srl.RunContext(env_config, rl_config, training=True)
    env.setup(context)
    [w.setup(context) for w in workers]
    trainer.setup(context)
    for episode in range(10000):
        step, rewards = _run_episode(env, workers, trainer, rendering=False)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")
    env.teardown()
    [w.teardown() for w in workers]
    trainer.teardown()

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    env.setup(context)
    [w.setup(context) for w in workers]
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, workers, None, rendering=False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")
    [w.teardown() for w in workers]
    env.teardown()

    # --- render
    context = srl.RunContext(env_config, rl_config, render_mode="terminal")
    env.setup(context)
    [w.setup(context) for w in workers]
    _run_episode(env, workers, None, rendering=True)
    [w.teardown() for w in workers]
    env.teardown()


def play_cpu(player_mode: str = "human"):
    import srl.rl.human

    env_config = srl.EnvConfig("OX")

    # make
    env = env_config.make()
    player = srl.make_worker(player_mode, env)
    cpu = env.make_worker("cpu")
    assert cpu is not None

    # set context
    context = srl.RunContext(env_config, render_mode="terminal")
    env.setup(context)
    player.setup(context)
    cpu.setup(context)

    # reset
    env.reset()
    player.reset(player_index=0)
    cpu.reset(player_index=1)

    while not env.done:
        # 1 step
        if env.next_player == 0:
            action = player.policy()
        else:
            action = cpu.policy()
        env.render()
        env.step(action)
        player.on_step()
        cpu.on_step()

    env.render()

    # teardown
    player.teardown()
    cpu.teardown()


if __name__ == "__main__":
    common.logger_print()

    main()
    print()
    print("-" * 20)
    print()
    play_cpu()
