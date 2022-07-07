from typing import Optional

import srl
from srl.base.env.base import EnvRun
from srl.base.env.singleplay_wrapper import SinglePlayEnvWrapper
from srl.base.rl.base import RLTrainer
from srl.base.rl.singleplay_wrapper import SinglePlayWorkerWrapper


def _run_episode(
    env: EnvRun,
    worker,
    trainer: Optional[RLTrainer],
    rendering=False,
):

    # change single play interface
    env = SinglePlayEnvWrapper(env)
    worker = SinglePlayWorkerWrapper(worker)

    # reset
    state = env.reset()
    worker.on_reset(env)

    if rendering:
        print("step 0")
        env.render()

    while not env.done:

        # action
        action = worker.policy(env)

        # render
        if rendering:
            worker.render(env)

        # step
        state, reward, done, env_info = env.step(action)
        work_info = worker.on_step(env)

        # train
        if trainer is None:
            train_info = {}
        else:
            train_info = trainer.train()

        # render
        if rendering:
            print(
                "step {}, action {}, reward: {}, done: {}, info: {} {} {}".format(
                    env.step_num, action, env.step_rewards[0], env.done, env_info, work_info, train_info
                )
            )
            env.render()

    return env.step_num, env.episode_rewards[0]


def main():

    env_config = srl.envs.Config("Grid")
    rl_config = srl.rl.ql.Config()

    # env init
    env = srl.envs.make(env_config)

    # rl init
    remote_memory, parameter, trainer, worker = srl.rl.make(rl_config, env)

    # --- train loop
    worker.set_play_info(training=True, distributed=False)
    for episode in range(10000):
        step, reward = _run_episode(env, worker, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    worker.set_play_info(training=False, distributed=False)
    step, reward = _run_episode(env, worker, trainer=None, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
