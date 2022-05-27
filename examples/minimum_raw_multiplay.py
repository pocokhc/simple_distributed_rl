import srl
from srl.base.rl.registration import make_worker


def _run_episode(
    env,
    workers,
    trainer,
    rendering=False,
):

    # --- reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    if rendering:
        print("step 0")
        env.render()

    while not env.done:

        # action
        actions = [w.policy(env) for w in workers]

        if rendering:
            for idx in env.next_player_indices:
                print(f"player {idx}")
                workers[idx].render(env)

        # step
        env_info = env.step(actions)
        worker_infos = [w.on_step(env) for w in workers]

        # --- trainer
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # --- render
        if rendering:
            print(
                "turn {}, actions {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, actions, env.step_rewards, env.done, env.next_player_indices, env_info
                )
            )
            for i in env.next_player_indices:
                print("player {} info: {}".format(i, worker_infos[i]))
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
        make_worker(srl.rl.random_play.Config(), env),
    ]

    # --- train loop
    workers[0].set_training(True, False)
    workers[1].set_training(False, False)
    for episode in range(10000):
        step, reward = _run_episode(env, workers, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    workers[0].set_training(False, False)
    workers[1].set_training(False, False)
    step, reward = _run_episode(env, workers, None, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
