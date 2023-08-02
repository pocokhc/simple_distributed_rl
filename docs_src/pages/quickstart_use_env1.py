import srl
from srl.algorithms import ql


def main():
    env_config = srl.EnvConfig("FrozenLake-v1")

    runner = srl.Runner(env_config, ql.Config())

    runner.render_terminal()


if __name__ == "__main__":
    main()
