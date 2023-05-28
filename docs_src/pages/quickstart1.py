from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    # create config
    config = runner.Config("Grid", ql.Config())

    # train
    parameter, _, _ = runner.train(config, timeout=10)

    # evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10)
    print(f"evaluate episodes: {rewards}")


if __name__ == "__main__":
    main()
