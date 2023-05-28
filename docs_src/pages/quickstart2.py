import os

import numpy as np

import srl
from srl import runner

# --- use env & algorithm load
# (Run "pip install gym pygame" to use the gym environment)
import gym  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip

# --- save parameter path
_parameter_path = "_params.dat"


# --- sample config
# For the parameters of Config, refer to the argument completion or the original code.
def _create_config():
    env_config = srl.EnvConfig("FrozenLake-v1")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)
    parameter = config.make_parameter()

    # --- Loads the file if it exists
    if os.path.isfile(_parameter_path):
        parameter.load(_parameter_path)

    return config, parameter


# --- train sample
def train():
    config, parameter = _create_config()

    if True:
        # sequence training
        parameter, remote_memory, history = runner.train(config, parameter=parameter, timeout=60)
    else:
        # distributed training
        parameter, remote_memory, history = runner.train_mp(config, parameter=parameter, timeout=60)

    # save parameter
    parameter.save(_parameter_path)


# --- evaluate sample
def evaluate():
    config, parameter = _create_config()
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")


# --- render sample
# You can watch the progress of 1 episode
def render():
    config, parameter = _create_config()
    runner.render(config, parameter)


# --- render window sample
#  (Run "pip install opencv-python pillow matplotlib pygame" to use the animation)
def render_window():
    config, parameter = _create_config()
    runner.render_window(config, parameter)


# --- animation sample
#  (Run "pip install opencv-python pillow matplotlib pygame" to use the animation)
def animation():
    config, parameter = _create_config()
    render = runner.animation(config, parameter)
    render.create_anime().save("_FrozenLake.gif")


if __name__ == "__main__":
    train()
    evaluate()
    render()
    render_window()
    animation()
