from typing import List, Optional

import srl
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLTrainer, WorkerRun
from srl.base.rl.registration import make_worker_rulebase


def _run_episode(
    env: EnvRun,
    workers: List[WorkerRun],
    trainer: Optional[RLTrainer],
    rendering: bool = False,
):

    # --- reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    if rendering:
        print("step 0")
        env.render()

    while not env.done:

        # action
        action = workers[env.next_player_index].policy(env)

        if rendering:
            print(f"player {env.next_player_index}")
            workers[env.next_player_index].render(env)

        # step
        env_info = env.step(action)
        worker_infos = [w.on_step(env) for w in workers]

        # --- trainer
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # --- render
        if rendering:
            print(
                "turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, action, env.step_rewards, env.done, env.next_player_index, env_info
                )
            )
            print("player {} info: {}".format(env.next_player_index, worker_infos[env.next_player_index]))
            print("train info: {}".format(train_info))
            env.render()

    return env.step_num, env.episode_rewards


def main():

    env_config = srl.envs.Config("OX")
    rl_config = srl.rl.ql.Config()

    # env init
    env = srl.envs.make(env_config)

    # rl init
    remote_memory, parameter, trainer, worker = srl.rl.make(rl_config, env)
    workers = [
        worker,
        make_worker_rulebase("random"),
    ]

    # --- train loop
    workers[0].set_play_info(training=True, distributed=False)
    workers[1].set_play_info(training=False, distributed=False)
    for episode in range(10000):
        step, reward = _run_episode(env, workers, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    workers[0].set_play_info(training=False, distributed=False)
    workers[1].set_play_info(training=False, distributed=False)
    step, reward = _run_episode(env, workers, None, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
