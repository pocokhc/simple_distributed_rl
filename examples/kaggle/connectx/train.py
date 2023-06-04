import os

import numpy as np
from model import create_config

from srl import runner
from srl.utils import common

common.logger_print()


path = os.path.join(os.path.dirname(__file__), "parameter.dat")


def train():
    config = create_config()

    # --- parameter
    parameter = config.make_parameter()
    if os.path.isfile(path):
        parameter.load(path)  # 以前学習したパラメータをロード

    # --- set players
    # config.players = [None, None]  # self play
    # config.players = [None, "random"]
    config.players = [None, "alphabeta8"]  # CPU

    # --- model summary
    config.model_summary(expand_nested=True)

    # --- train
    if True:
        # sequence training
        parameter, memory, history = runner.train(config, parameter=parameter, timeout=60 * 60 * 5)
    else:
        # distributed training
        config.actor_num = 2
        parameter, memory, history = runner.train_mp(config, init_parameter=parameter, timeout=60 * 60 * 5)

    # --- save parameter
    parameter.save(path)


def eval():
    config = create_config()
    parameter = config.make_parameter()
    parameter.load(path)

    for players in [
        [None, None],
        [None, "random"],
        ["random", None],
        [None, "alphabeta7"],
        ["alphabeta7", None],
    ]:
        config.players = players
        rewards = runner.evaluate(config, parameter, max_episodes=5)
        print(f"{np.mean(rewards, axis=0)}, {players}")


def view():
    config = create_config()
    parameter = config.make_parameter()
    parameter.load(path)

    config.players = [None, "human"]
    runner.render_terminal(config, parameter=parameter)


if __name__ == "__main__":
    train()
    eval()
    view()
