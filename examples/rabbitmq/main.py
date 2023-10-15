import srl
from srl.algorithms import ql


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    runner = srl.Runner(env_config, rl_config)
    runner.train_rabbitmq("127.0.0.1", max_train_count=100_000)

    print(runner.evaluate())


if __name__ == "__main__":
    main()
