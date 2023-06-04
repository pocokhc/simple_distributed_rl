import numpy as np

import srl
from srl import runner

# --- env & algorithm load
import srl.envs.ox  # isort: skip # noqa F401
from srl.algorithms import mcts  # isort: skip


def main():
    env_config = srl.EnvConfig("OX")
    rl_config = mcts.Config()
    config = runner.Config(env_config, rl_config)

    # --- train
    config.players = [None, "random"]
    parameter, _, _ = runner.train(config, max_episodes=10000)

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
    ]:
        config.players = players
        rewards = runner.evaluate(config, parameter, max_episodes=100)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")

    # --- rendering
    config.players = [None, "cpu"]
    runner.render_terminal(config, parameter)

    # --- 対戦
    config.players = [None, "human"]
    runner.render_terminal(config, parameter)


if __name__ == "__main__":
    main()
