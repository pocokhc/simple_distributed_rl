from typing import List, Optional

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
        action = workers[env.next_player].policy()

        # --- worker render
        if rendering:
            print(f"player {env.next_player}")
            workers[env.next_player].render()

        # 3. step
        env.step(action)
        [w.on_step() for w in workers]

        # --- env render
        if rendering:
            print(
                "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, action, env.rewards, env.done, env.next_player, env.info
                )
            )
            print("player {} info: {}".format(env.next_player, workers[env.next_player].info))
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
    env.setup()
    context = srl.RunContext(env_config, rl_config, training=True)
    [w.on_start(context) for w in workers]
    trainer.on_start(context)
    for episode in range(10000):
        step, rewards = _run_episode(env, workers, trainer, rendering=False)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")
    [w.on_end() for w in workers]
    trainer.on_end()

    # --- evaluate
    env.setup()
    context = srl.RunContext(env_config, rl_config)
    [w.on_start(context) for w in workers]
    trainer.on_start(context)
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, workers, None, rendering=False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")
    [w.on_end() for w in workers]
    trainer.on_end()

    # --- render
    context = srl.RunContext(env_config, rl_config, render_mode="terminal")
    env.setup(context.render_mode)
    [w.on_start(context) for w in workers]
    _run_episode(env, workers, None, rendering=True)
    [w.on_end() for w in workers]


def play_cpu():
    import srl.rl.human

    env_config = srl.EnvConfig("OX")

    # make
    env = env_config.make()
    player = srl.make_worker("human", env)
    cpu = env.make_worker("cpu")
    assert cpu is not None

    # set context
    context = srl.RunContext(env_config, render_mode="terminal")
    env.setup(context.render_mode)
    player.on_start(context)
    cpu.on_start(context)

    # reset
    env.reset()
    player.on_reset(player_index=0)
    cpu.on_reset(player_index=1)

    env.render()

    while not env.done:
        # 1 step
        if env.next_player == 0:
            action = player.policy()
        else:
            action = cpu.policy()
        env.step(action)
        player.on_step()
        cpu.on_step()

        env.render()

    # end
    player.on_end()
    cpu.on_end()


if __name__ == "__main__":
    common.logger_print()

    main()
    play_cpu()
