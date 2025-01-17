import os

from srl_model import create_runner

from srl.utils import common

common.logger_print()
param_path = os.path.join(os.path.dirname(__file__), "_parameter.dat")


def train():
    runner = create_runner()

    # --- 学習を実施
    runner.train(timeout=5)

    # --- save parameter
    runner.save_parameter(param_path)


def eval():
    runner = create_runner()
    runner.load_parameter(param_path)

    rewards = runner.evaluate(max_episodes=50)
    print(f"{rewards}")


if __name__ == "__main__":
    train()
    eval()
