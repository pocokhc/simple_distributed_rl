import numpy as np

import srl
from srl.algorithms import ql


def main():
    runner = srl.Runner("Grid", ql.Config())
    runner.train_remote(max_train_count=10_000)

    # result
    rewards = runner.evaluate()
    print(np.mean(rewards))


if __name__ == "__main__":
    main()
