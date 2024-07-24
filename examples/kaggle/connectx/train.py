import os

import numpy as np
from model import create_runner

from srl.utils import common

common.logger_print()


path = os.path.join(os.path.dirname(__file__), "parameter.dat")


def train():
    runner = create_runner()

    if os.path.isfile(path):
        runner.load_parameter(path)  # 以前学習したパラメータをロード

    # --- players
    players = [None, None]  # self play
    # players = [None, "random"]
    # players = [None, "alphabeta8"]

    # --- train
    runner.train(timeout=10, players=players)  # change time: timeout=60 * 60 * 5

    # --- save parameter
    runner.save_parameter(path)


def eval():
    runner = create_runner()
    runner.load_parameter(path)

    for players in [
        [None, None],
        [None, "random"],
        ["random", None],
        [None, "alphabeta7"],
        ["alphabeta7", None],
    ]:
        rewards = runner.evaluate(max_episodes=5, players=players)
        print(f"{np.mean(rewards, axis=0)}, {players}")


def view():
    runner = create_runner()
    runner.load_parameter(path)
    runner.render_terminal(players=[None, "human"])


if __name__ == "__main__":
    train()
    eval()
    view()
