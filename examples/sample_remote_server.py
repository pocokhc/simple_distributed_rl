import numpy as np

from srl import runner
from srl.algorithms import ql
from srl.envs import grid  # noqa F401


def main():
    config = runner.Config("Grid", ql.Config(), actor_num=1)

    # run server
    parameter, _, _ = runner.train_remote(config, max_train_count=10_000)

    # result
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(np.mean(rewards))


if __name__ == "__main__":
    main()
