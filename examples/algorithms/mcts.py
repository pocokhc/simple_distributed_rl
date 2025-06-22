import numpy as np

import srl
from srl.algorithms import mcts
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = mcts.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- train
    runner.train(max_episodes=10000, players=[None, "random"])

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
    ]:
        rewards = runner.evaluate(max_episodes=100, players=players)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")

    # --- rendering
    runner.render_terminal(players=[None, "cpu"])

    # --- 対戦
    runner.render_terminal(players=[None, "human"])


if __name__ == "__main__":
    main()
