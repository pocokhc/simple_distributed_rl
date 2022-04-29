import srl
from srl.base.env.single_play_wrapper import SinglePlayerWrapper


def _run_episode(
    env,
    worker,
    trainer,
    training,
    rendering=False,
):
    worker.set_training(training)
    env = SinglePlayerWrapper(env)  # change single play interface

    state, invalid_actions = env.reset()

    done = False
    step = 0
    total_reward = 0

    worker.on_reset(state, invalid_actions, env)

    if rendering:
        print("step 0")
        env.render()

    while True:

        # render
        if rendering:
            worker.render(env)

        # action
        action = worker.policy(state, invalid_actions, env)

        # env step
        state, reward, done, invalid_actions, env_info = env.step(action)
        step += 1
        total_reward += reward

        if step > env.max_episode_steps:
            done = True

        # rl step
        work_info = worker.on_step(state, reward, done, invalid_actions, env)

        # train
        if training and trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # render
        if rendering:
            print(
                "step {}, action {}, reward: {}, done: {}, info: {} {} {}".format(
                    step, action, reward, done, env_info, work_info, train_info
                )
            )
            env.render()

        # step after
        if done:
            break

    return step, total_reward


def main():

    env_config = srl.envs.Config("Grid")
    rl_config = srl.rl.ql.Config()

    # check rl_config
    rl_config.assert_params()

    # env init
    env = srl.envs.make(env_config, rl_config)

    # rl init
    remote_memory, parameter, trainer, worker = srl.rl.make(rl_config, env)

    # --- train loop
    for episode in range(10000):
        step, reward = _run_episode(env, worker, trainer, training=True)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    step, reward = _run_episode(env, worker, None, training=False, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
