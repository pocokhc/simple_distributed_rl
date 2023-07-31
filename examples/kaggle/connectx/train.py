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

    # --- set players
    # runner.set_players([None, None])  # self play
    # runner.set_players([None, "random"])
    runner.set_players([None, "alphabeta8"])  # CPU

    # --- train
    if True:
        # sequence training
        runner.train(timeout=60 * 60 * 5)
    else:
        # distributed training
        runner.train_mp(actor_num=2, timeout=60 * 60 * 5)

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
        runner.set_players(players)
        rewards = runner.evaluate(max_episodes=5)
        print(f"{np.mean(rewards, axis=0)}, {players}")


def view():
    runner = create_runner()
    runner.load_parameter(path)

    runner.set_players([None, "human"])
    runner.render_terminal()


if __name__ == "__main__":
    train()
    eval()
    view()
