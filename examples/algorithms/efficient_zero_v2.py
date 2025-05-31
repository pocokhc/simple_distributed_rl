import os
from pathlib import Path

import srl
from srl.algorithms import efficient_zero_v2
from srl.utils import common

common.logger_print()


def create_runner():
    env_config = srl.EnvConfig("EasyGrid-layer")

    rl_config = efficient_zero_v2.Config()
    rl_config.set_small_params()
    rl_config.num_simulations = 20
    rl_config.lr = 0.001
    rl_config.reward_loss_coeff = 1.0
    rl_config.consistency_loss_coeff = 1
    rl_config.unroll_steps = 2
    rl_config.memory.warmup_size = 100

    runner = srl.Runner(env_config, rl_config)
    return runner


def train():
    runner = create_runner()
    runner.train_mp(max_train_count=5000)
    runner.save_parameter(str(Path(__file__).parent / "_efficient_zero.dat"))


def eval():
    runner = create_runner()
    runner.load_parameter(str(Path(__file__).parent / "_efficient_zero.dat"))

    print(runner.evaluate())

    path = os.path.join(str(Path(__file__).parent / "_efficient_zero.gif"))
    runner.animation_save_gif(path)


if __name__ == "__main__":
    train()
    eval()
