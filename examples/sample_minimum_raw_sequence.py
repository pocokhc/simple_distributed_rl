import gym
from srl import rl
from srl.base.rl.env_for_rl import EnvForRL


def _run_episode(
    env,
    worker,
    trainer,
    rendering=False,
):
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    valid_actions = env.fetch_valid_actions()
    worker.on_reset(state, valid_actions)

    for step in range(env.max_episode_steps):

        # action
        env_action, worker_action = worker.policy(state, valid_actions)
        if valid_actions is not None:
            assert env_action in valid_actions

        # env step
        next_state, reward, done, env_info = env.step(env_action)
        step += 1
        total_reward += reward
        next_valid_actions = env.fetch_valid_actions()

        # rl step
        work_info = worker.on_step(state, worker_action, next_state, reward, done, valid_actions, next_valid_actions)
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # render
        if rendering:
            env.render()
            worker.render(state, valid_actions, env.action_to_str)
            print(
                "{} action {}, reward: {}, done: {}, info: {} {} {}".format(
                    step, env_action, reward, done, env_info, work_info, train_info
                )
            )

        # step after
        if done:
            break
        state = next_state
        valid_actions = next_valid_actions

    return step, total_reward


def main():

    # parameter
    env_name = "FrozenLake-v1"
    rl_config = rl.ql.Config()

    # init
    rl_config.assert_params()
    env = EnvForRL(gym.make(env_name), rl_config)
    rl_module = rl.make(rl_config.getName())
    parameter = rl_module.Parameter(rl_config)
    memory = rl_module.RemoteMemory(rl_config)
    trainer = rl_module.Trainer(rl_config, parameter, memory)
    worker = rl_module.Worker(rl_config, parameter, memory, 0)

    # --- train loop
    worker.set_training(True)
    for episode in range(10000):
        step, reward = _run_episode(env, worker, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    worker.set_training(False)
    step, reward = _run_episode(env, worker, None, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
