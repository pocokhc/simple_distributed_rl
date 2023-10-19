import srl
from srl.algorithms import ql
from srl.utils import common


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    runner = srl.Runner(env_config, rl_config)
    runner.train_rabbitmq("127.0.0.1", timeout=30)

    print(runner.evaluate())


if __name__ == "__main__":
    common.logger_print()
    main()
