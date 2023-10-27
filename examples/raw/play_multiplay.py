from typing import List, Optional

import numpy as np

import srl
import srl.rl.random_play
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils import common

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    workers: List[WorkerRun],
    trainer: Optional[RLTrainer],
):
    training = trainer is not None

    # 1. reset
    env.reset()
    [w.on_reset(player_index=i, training=training) for i, w in enumerate(workers)]

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy()

        # 3. step
        env.step(action)
        [w.on_step() for w in workers]

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

    return env.step_num, env.episode_rewards


def _render(env: EnvRun, workers: List[WorkerRun]):
    # 1. reset
    # (Only one of the window mode)
    env.reset(render_mode="terminal")
    [w.on_reset(i, training=False, render_mode="terminal") for i, w in enumerate(workers)]

    # --- render
    print("step 0")
    env.render()

    while not env.done:
        # 2. action
        action = workers[env.next_player_index].policy()

        # --- worker render
        print(f"player {env.next_player_index}")
        workers[env.next_player_index].render()

        # 3. step
        env.step(action)
        [w.on_step() for w in workers]

        # --- env render
        print(
            "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                env.step_num, action, env.step_rewards, env.done, env.next_player_index, env.info
            )
        )
        print("player {} info: {}".format(env.next_player_index, workers[env.next_player_index].info))
        env.render()

    print(f"step: {env.step_num}, reward: {env.episode_rewards}")


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # --- make instance
    env = srl.make_env(env_config)
    rl_config.setup(env)
    parameter = srl.make_parameter(rl_config)
    remote_memory = srl.make_memory(rl_config)
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)
    workers = [
        srl.make_worker(rl_config, env, parameter, remote_memory),
        srl.make_worker_rulebase("random", env),
    ]

    # --- train loop
    for episode in range(10000):
        step, rewards = _run_episode(env, workers, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")

    # --- evaluate
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, workers, None)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")

    # --- render
    _render(env, workers)


def play_cpu():
    import srl.rl.human

    env_config = srl.EnvConfig("OX")

    # make
    env = srl.make_env(env_config)
    player = srl.make_worker_rulebase("human", env)
    cpu = env.make_worker("cpu")
    assert cpu is not None

    # reset
    env.reset(render_mode="terminal")
    player.on_reset(player_index=0, training=False)
    cpu.on_reset(player_index=1, training=False)

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
