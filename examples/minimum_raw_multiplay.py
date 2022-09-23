from typing import List, Optional

import srl
import srl.rl.random_play
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_worker_rulebase
from srl.base.rl.worker import WorkerRun

# --- env & algorithm
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def _run_episode(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: Optional[RLRemoteMemory],
    training: bool,
    rendering: bool = False,
):

    workers: List[WorkerRun] = [
        srl.make_worker(rl_config, parameter, remote_memory, training=training, distributed=False),
        make_worker_rulebase("random"),
    ]
    if training:
        trainer = srl.make_trainer(rl_config, parameter, remote_memory)
    else:
        trainer = None

    # --- set render mode
    if rendering:
        env.set_render_mode("terminal")
        [w.set_render_mode("terminal") for w in workers]

    # --- reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    if rendering:
        print("step 0")
        env.render()

    while not env.done:

        # --- action
        action = workers[env.next_player_index].policy(env)

        # worker render
        if rendering:
            print(f"player {env.next_player_index}")
            workers[env.next_player_index].render(env)

        # --- step
        env.step(action)
        [w.on_step(env) for w in workers]

        # --- trainer
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # env render
        if rendering:
            print(
                "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, action, env.step_rewards, env.done, env.next_player_index, env.info
                )
            )
            print("player {} info: {}".format(env.next_player_index, workers[env.next_player_index].info))
            print("train info: {}".format(train_info))
            env.render()

    return env.step_num, env.episode_rewards


def main():

    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()

    # env init
    env = srl.make_env(env_config)

    # rl init
    rl_config.reset_config(env)
    parameter = srl.make_parameter(rl_config)
    remote_memory = srl.make_remote_memory(rl_config)

    # --- train loop
    for episode in range(10000):
        step, reward = _run_episode(env, rl_config, parameter, remote_memory, training=True)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    step, reward = _run_episode(env, rl_config, parameter, None, training=False, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
