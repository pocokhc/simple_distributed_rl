import os

import numpy as np

import srl
from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import planet  # isort: skip

_parameter_path = os.path.join(os.path.dirname(__file__), "_planet_param.dat")


def _create_config():
    rl_config = planet.Config(
        deter_size=50,
        stoch_size=10,
        num_units=100,
        cnn_depth=32,
        batch_size=8,
        batch_length=20,
        lr=0.001,
        free_nats=3.0,
        kl_scale=1.0,
        enable_overshooting_loss=False,
        # GA
        pred_action_length=10,
        num_generation=20,
        num_individual=5,
        num_simulations=5,
        print_ga_debug=False,
        # action_algorithm="random",
    )
    rl_config.change_observation_render_image = True
    rl_config.parameter_path = _parameter_path
    env_config = srl.EnvConfig("EasyGrid")
    env_config.max_episode_steps = 21
    return runner.Config(env_config, rl_config), rl_config


def train():
    config, rl_config = _create_config()
    config.model_summary()

    # train
    _, memory, _ = runner.train(
        config,
        max_episodes=1000,
        disable_trainer=True,
    )
    rl_config.memory_warmup_size = rl_config.batch_size + 1
    parameter, memory, history = runner.train_only(
        config,
        remote_memory=memory,
        max_train_count=30_000,
    )
    parameter.save(_parameter_path)


def evaluate():
    config, rl_config = _create_config()
    rewards = runner.evaluate(config, max_episodes=10, print_progress=True)
    print(rewards)
    print("mean", np.mean(rewards))


def animation():
    config, rl_config = _create_config()
    render = runner.animation(config, print_progress=True)
    render.create_anime(draw_info=True).save(os.path.join(os.path.dirname(__file__), "_planet.gif"))


if __name__ == "__main__":
    train()
    evaluate()
    animation()
