import numpy as np
import srl
from srl.base.rl.registration import make_worker


def _run_episode(
    env,
    workers,
    trainer,
    rendering=False,
):
    # --- env
    state, next_player_indices = env.reset()

    done = False
    step = 0
    total_rewards = np.zeros(env.player_num)

    if rendering:
        print("step 0")
        env.render()

    # --- players
    players_status = ["INIT" for _ in range(env.player_num)]
    players_step_reward = np.zeros(env.player_num)
    worker_info_list = [None for _ in range(env.player_num)]

    while True:

        # --- rl before step
        actions = []
        for idx in next_player_indices:

            # --- rl init
            if players_status[idx] == "INIT":
                workers[idx].on_reset(state, idx, env)
                players_status[idx] = "RUNNING"

            if rendering:
                print(f"player {idx}")
                workers[idx].render(env)

            # --- rl action
            action = workers[idx].policy(state, idx, env)
            actions.append(action)

        # --- env step
        state, rewards, done, next_player_indices, env_info = env.step(actions)
        step += 1

        # update reward
        rewards = np.asarray(rewards)
        total_rewards += rewards
        players_step_reward += rewards

        # done
        if step > env.max_episode_steps:
            done = True

        # --- rl after step
        if done:
            # 終了の場合は全playerを実行
            next_player_indices = [i for i in range(env.player_num)]
        for idx in next_player_indices:
            if players_status[idx] != "RUNNING":
                continue
            worker_info_list[idx] = workers[idx].on_step(
                state,
                players_step_reward[idx],
                done,
                idx,
                env,
            )
            players_step_reward[idx] = 0

        # --- trainer
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # --- render
        if rendering:
            print(
                "turn {}, actions {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    step, actions, rewards, done, next_player_indices, env_info
                )
            )
            for i in next_player_indices:
                print("player {} info: {}".format(i, worker_info_list[i]))
            print("train info: {}".format(train_info))
            env.render()

        # step after
        if done:
            break

    return step, total_rewards


def main():

    env_config = srl.envs.Config("OX")
    rl_config = srl.rl.ql.Config()

    # check rl_config
    rl_config.assert_params()

    # env init
    env = srl.envs.make(env_config, rl_config)

    # rl init
    remote_memory, parameter, trainer, worker = srl.rl.make(rl_config, env)
    workers = [
        worker,
        make_worker(srl.rl.random_play.Config(), env),
    ]

    # --- train loop
    workers[0].set_training(True)
    workers[1].set_training(False)
    for episode in range(10000):
        step, reward = _run_episode(env, workers, trainer)
        if episode % 1000 == 0:
            print(f"{episode} / 10000 episode, {step} step, {reward} reward")

    # --- render
    workers[0].set_training(False)
    workers[1].set_training(False)
    step, reward = _run_episode(env, workers, None, rendering=True)
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
