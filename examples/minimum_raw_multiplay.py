import numpy as np
import srl
from srl.base.env.env_for_rl import EnvForRL
from srl.base.rl.registration import make_worker
from srl.rl.processor import ContinuousProcessor, DiscreteProcessor, ObservationBoxProcessor


def _run_episode(
    env,
    workers,
    trainer,
    rendering=False,
):
    # --- env
    states, player_indexes = env.reset()
    done = False
    step = 0
    total_rewards = np.zeros(env.player_num)
    invalid_actions_list = env.fetch_invalid_actions()
    env_actions = [None for _ in range(env.player_num)]

    if rendering:
        print("step 0")
        env.render()

    # --- players
    players = [
        {
            "status": "INIT",
            "state": None,
            "action": None,
            "invalid_actions": None,
            "reward": 0,
            "worker": workers[i],
        }
        for i in range(env.player_num)
    ]
    work_info_list = [None for _ in range(env.player_num)]

    while True:

        # --- rl before step
        for i in player_indexes:
            worker = players[i]["worker"]

            # --- rl init
            if players[i]["status"] == "INIT":
                worker.on_reset(states[i], invalid_actions_list[i], env, player_indexes)
                players[i]["status"] = "RUNNING"

            if rendering:
                print(f"player {i}")
                players[i]["worker"].render(states[i], invalid_actions_list[i], env)

            # --- rl action
            env_action, worker_action = worker.policy(states[i], invalid_actions_list[i], env, player_indexes)
            assert env_action not in invalid_actions_list[i]

            env_actions[i] = env_action
            players[i]["state"] = states[i]
            players[i]["action"] = worker_action
            players[i]["invalid_actions"] = invalid_actions_list[i]
            players[i]["reward"] = 0

        # --- env step
        next_states, rewards, next_player_indexes, done, env_info = env.step(env_actions, player_indexes)
        step += 1
        total_rewards += np.asarray(rewards)
        next_invalid_actions_list = env.fetch_invalid_actions()

        # update reward
        for i in range(len(rewards)):
            players[i]["reward"] += rewards[i]

        # done
        if step > env.max_episode_steps:
            done = True

        # --- rl after step
        if done:
            # 終了の場合は全playerを実行
            next_player_indexes = [i for i in range(env.player_num)]
        for i in next_player_indexes:
            if players[i]["status"] == "RUNNING":
                work_info = players[i]["worker"].on_step(
                    players[i]["state"],
                    players[i]["action"],
                    next_states[i],
                    players[i]["reward"],
                    done,
                    players[i]["invalid_actions"],
                    next_invalid_actions_list[i],
                    env,
                )
                work_info_list[i] = work_info

        # --- trainer
        if trainer is not None:
            train_info = trainer.train()
        else:
            train_info = {}

        # --- render
        if rendering:
            print(
                "turn {}, player {}, rewards: {}, done: {}, info: {}, ".format(
                    step, player_indexes, rewards, done, env_info
                )
            )
            for i in player_indexes:
                print("player {} action {}, info: {}".format(i, env_actions[i], work_info_list[i]))
            print("train info: {}".format(train_info))
            env.render()

        # step after
        if done:
            break
        states = next_states
        invalid_actions_list = next_invalid_actions_list
        player_indexes = next_player_indexes

    return step, total_rewards


def main():

    env_name = "OX"
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
