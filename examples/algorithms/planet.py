import os
from typing import cast

import numpy as np

import srl
from srl.algorithms import planet

_parameter_path = os.path.join(os.path.dirname(__file__), "_planet_param.dat")


def _create_runner():
    rl_config = planet.Config(
        deter_size=50,
        stoch_size=10,
        num_units=100,
        cnn_depth=32,
        batch_size=16,
        batch_length=11,
        lr=0.001,
        free_nats=0.1,
        kl_scale=1.0,
        enable_overshooting_loss=False,
        # GA
        pred_action_length=5,
        num_generation=20,
        num_individual=5,
        num_simulations=5,
        print_ga_debug=False,
        # action_algorithm="random",
    )
    rl_config.use_render_image_for_observation = True
    rl_config.parameter_path = _parameter_path
    env_config = srl.EnvConfig("EasyGrid")
    env_config.max_episode_steps = 10
    return srl.Runner(env_config, rl_config)


def train():
    runner = _create_runner()
    rl_config = cast(planet.Config, runner.rl_config)

    runner.model_summary()

    # train
    runner.train(max_episodes=1000, disable_trainer=True)

    rl_config.memory.warmup_size = rl_config.batch_size + 1
    runner.train_only(max_train_count=2_000)

    runner.save(_parameter_path)


def evaluate():
    runner = _create_runner()
    rewards = runner.evaluate(max_episodes=10)
    print(rewards)
    print("mean", np.mean(rewards))


def animation():
    runner = _create_runner()

    path = os.path.join(os.path.dirname(__file__), "_planet.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    train()
    evaluate()
    animation()
