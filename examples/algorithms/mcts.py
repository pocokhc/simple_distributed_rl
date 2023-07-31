import numpy as np

import srl
from srl.algorithms import mcts


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = mcts.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- train
    runner.set_players([None, "random"])
    runner.train(max_episodes=10000)

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
    ]:
        runner.set_players(players)
        rewards = runner.evaluate(max_episodes=100)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")

    # --- rendering
    runner.set_players([None, "cpu"])
    runner.render_terminal()

    # --- 対戦
    runner.set_players([None, "human"])
    runner.render_terminal()


if __name__ == "__main__":
    main()
