import numpy as np

import srl
from srl import runner

# --- env & algorithm load
# (Run "pip install gymnasium pygame" to use the gymnasium environment)
import gymnasium  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip

# --- save parameter path
_parameter_path = "_params.dat"


# --- sample config
# For the parameters of Config, refer to the argument completion or the original code.
def _create_config(load_parameter: bool):
    env_config = srl.EnvConfig("FrozenLake-v1")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)
    parameter = config.make_parameter()

    # --- Loads the file
    if load_parameter:
        parameter.load(_parameter_path)

    return config, parameter


# --- train sample
def train():
    config, _ = _create_config(load_parameter=False)

    if True:
        # sequence training
        parameter, remote_memory, history = runner.train(config, timeout=10)
    else:
        # distributed training
        parameter, remote_memory, history = runner.train_mp(config, timeout=10)

    # save parameter
    parameter.save(_parameter_path)


# --- evaluate sample
def evaluate():
    config, parameter = _create_config(load_parameter=True)
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


# --- render terminal sample
def render_terminal():
    config, parameter = _create_config(load_parameter=True)
    runner.render_terminal(config, parameter)


# --- render window sample
#  (Run "pip install pillow pygame" to use the render_window)
def render_window():
    config, parameter = _create_config(load_parameter=True)
    runner.render_window(config, parameter)


# --- animation sample
#  (Run "pip install opencv-python pillow pygame" to use the animation)
def animation():
    config, parameter = _create_config(load_parameter=True)
    render = runner.animation(config, parameter)
    render.create_anime().save("_FrozenLake.gif")


# --- replay window sample
#  (Run "pip install opencv-python pillow pygame" to use the replay_window)
def replay_window():
    config, parameter = _create_config(load_parameter=True)
    runner.replay_window(config, parameter)


if __name__ == "__main__":
    train()
    evaluate()
    render_terminal()
    render_window()
    animation()
    replay_window()
