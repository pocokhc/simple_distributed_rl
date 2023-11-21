import srl
from srl.algorithms import ql
from srl.runner.distribution import RedisParameters
from srl.utils import common


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    runner = srl.Runner(env_config, rl_config)

    runner.train_distribution(
        RedisParameters(host="localhost"),
        max_train_count=1000,
        progress_interval=10,
    )

    print(runner.evaluate())


if __name__ == "__main__":
    common.logger_print()
    main()
