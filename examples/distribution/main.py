import srl
from srl.algorithms import ql
from srl.runner.distribution import ServerParameters
from srl.utils import common


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    runner = srl.Runner(env_config, rl_config)

    params = ServerParameters(redis_host="localhost", rabbitmq_host="localhost")
    runner.train_distribution(params, max_train_count=1000)

    print(runner.evaluate())


if __name__ == "__main__":
    common.logger_print()
    main()
