import os

import srl
from srl.algorithms import muzero
from srl.utils import common

common.logger_print()


def create_runner():
    rl_config = muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=32,
        lr=0.001,
        reward_range=(-2, 2),
        reward_range_num=10,
        value_range=(-10, 10),
        value_range_num=100,
        unroll_steps=1,
        dynamics_blocks=1,
        enable_rescale=False,
        weight_decay=0,
    )
    rl_config.memory.warmup_size = 100
    rl_config.memory.capacity = 100_000
    rl_config.input_image_block.set_alphazero_block(1, 64)
    env_config = srl.EnvConfig("EasyGrid-layer")
    runner = srl.Runner(env_config, rl_config)
    return runner


def main():
    runner = create_runner()

    # --- train
    runner.train_mp(max_train_count=10000)
    runner.save_parameter(os.path.join(os.path.dirname(__file__), "_muzero.dat"))


def eval():
    runner = create_runner()
    runner.load_parameter(os.path.join(os.path.dirname(__file__), "_muzero.dat"))
    runner.model_summary()

    # --- evaluate
    rewards = runner.evaluate(max_episodes=2)
    print("mean", rewards)

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_muzero.gif")
    runner.animation_save_gif(path)

    runner.play_window()


if __name__ == "__main__":
    main()
    eval()
