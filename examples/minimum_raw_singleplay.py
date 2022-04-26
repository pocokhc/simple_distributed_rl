import srl.envs
import srl.rl
from srl.base.env.env_for_rl import EnvForRL
from srl.base.env.singleplay_wrapper import SinglePlayerWrapper
from srl.rl.processor import ContinuousProcessor, DiscreteProcessor, ObservationBoxProcessor


def _run_episode(
    env,
    worker,
    trainer,
    training,
    rendering=False,
):
    worker.set_training(training)
    env = SinglePlayerWrapper(env)  # change single play interface

    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    invalid_actions = env.fetch_invalid_actions()
    worker.on_reset(state, invalid_actions, env, [0])

    if rendering:
        print("step 0")
        env.render()

    while True:

        # render
        if rendering:
            worker.render(state, invalid_actions, env)

        # action
        env_action, worker_action = worker.policy(state, invalid_actions, env, [0])
        assert env_action not in invalid_actions

        # env step
        next_state, reward, done, env_info = env.step(env_action)
        step += 1
        total_reward += reward
        next_invalid_actions = env.fetch_invalid_actions()

        if step > env.max_episode_steps:
            done = True

        # rl step
        work_info = worker.on_step(
            state, worker_action, next_state, reward, done, invalid_actions, next_invalid_actions, env
        )

        # train
        if training and trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # render
        if rendering:
            print(
                "step {}, action {}, reward: {}, done: {}, info: {} {} {}".format(
                    step, env_action, reward, done, env_info, work_info, train_info
                )
            )
            env.render()

        # step after
        if done:
            break
        state = next_state
        invalid_actions = next_invalid_actions

    return step, total_reward


def main():

    env_name = "Grid"
    rl_config = srl.rl.ql.Config()

    # env processors
    processors = [
        ObservationBoxProcessor(),
        DiscreteProcessor(),
        ContinuousProcessor(),
    ]

    rl_config.assert_params()

    # env init
    env = srl.envs.make(env_name)
    env = EnvForRL(env, rl_config, processors=processors)

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
