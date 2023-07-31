import srl
from srl.algorithms import ql  # algorithm load


def main():
    # create Runner
    runner = srl.Runner("Grid", ql.Config())

    # train
    runner.train(timeout=10)

    # evaluate
    rewards = runner.evaluate()
    print(f"evaluate episodes: {rewards}")


if __name__ == "__main__":
    main()
