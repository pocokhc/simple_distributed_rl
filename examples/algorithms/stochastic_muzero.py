import os
from pathlib import Path

import srl
from srl.algorithms import stochastic_muzero
from srl.utils import common

common.logger_print()


def create_runner():
    rl_config = stochastic_muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        reward_range=(-2, 2),
        reward_range_num=10,
        value_range=(-2, 2),
        value_range_num=10,
        unroll_steps=2,
        dynamics_blocks=1,
        enable_rescale=False,
        codebook_size=4,
    )
    rl_config.lr = 0.01
    rl_config.lr_scheduler.set_step(10_000, 0.001)
    rl_config.memory.capacity = 100_000
    rl_config.memory.warmup_size = 100
    rl_config.input_image_block.set_alphazero_block(1, 16)

    env_config = srl.EnvConfig("Grid-layer")
    return srl.Runner(env_config, rl_config)


def train():
    runner = create_runner()

    runner.train_mp(max_train_count=5000)
    runner.save_parameter(str(Path(__file__).parent / "_stochastic_muzero.dat"))


def eval():
    runner = create_runner()
    runner.load_parameter(str(Path(__file__).parent / "_stochastic_muzero.dat"))
    runner.model_summary()

    print(runner.evaluate())

    path = os.path.join(str(Path(__file__).parent / "_stochastic_muzero.gif"))
    runner.animation_save_gif(path)


if __name__ == "__main__":
    train()
    eval()
