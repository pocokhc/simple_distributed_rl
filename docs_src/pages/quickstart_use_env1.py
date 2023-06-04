import srl
from srl import runner
from srl.algorithms import ql  # load algorithm


def main():
    env_config = srl.EnvConfig("FrozenLake-v1")

    config = runner.Config(env_config, ql.Config())

    runner.render_terminal(config)


if __name__ == "__main__":
    main()
