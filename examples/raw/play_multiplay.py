from typing import List

import numpy as np

import srl
import srl.rl.random_play
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter
from srl.base.rl.worker import WorkerRun

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
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

    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter, remote_memory, training=training, distributed=False),
        srl.make_worker_rulebase("random", training=training, distributed=False),
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


def _render(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
):
    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter),
        srl.make_worker_rulebase("random"),
    ]

    # 1. reset
    env.reset(render_mode="terminal")
    [w.on_reset(env, i, render_mode="terminal") for i, w in enumerate(workers)]

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

    print(f"step: {env.step_num}, reward: {env.episode_rewards}")


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # env init
    env = srl.make_env(env_config)
    rl_config.reset(env)

    # make parameter
    parameter = srl.make_parameter(rl_config, env)

    # --- train loop
    for episode in range(10000):
        step, rewards = _run_episode(env, rl_config, parameter, True)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {rewards} reward")

    # --- evaluate
    rewards_list = []
    for episode in range(100):
        _, rewards = _run_episode(env, rl_config, parameter, False)
        rewards_list.append(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards_list, axis=0)}")

    # --- render
    _render(env, rl_config, parameter)


def play_cpu():
    import srl.rl.human

    env_config = srl.EnvConfig("OX")

    # make
    env = srl.make_env(env_config)
    player = srl.make_worker_rulebase("human")
    cpu = env.make_worker("cpu")
    assert cpu is not None

    # reset
    env.reset(render_mode="terminal")
    player.on_reset(env, player_index=0)
    cpu.on_reset(env, player_index=1)

    env.render()

    while not env.done:
        # 1 step
        if env.next_player_index == 0:
            action = player.policy(env)
        else:
            action = cpu.policy(env)
        env.step(action)
        player.on_step(env)
        cpu.on_step(env)

        env.render()

    env.close()


if __name__ == "__main__":
    main()
    # play_cpu()
