import numpy as np

import srl
from srl.algorithms import ql  # algorithm load
from srl.utils import common

common.logger_print()

# --- save parameter path
_parameter_path = "_params.dat"


# --- sample config
# For the parameters of Config, refer to the argument completion or the original code.
def _create_runner(load_parameter: bool):
    # (Run "pip install gymnasium pygame" to use the gymnasium environment)
    env_config = srl.EnvConfig("FrozenLake-v1")

    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- load parameter
    if load_parameter:
        runner.load_parameter(_parameter_path)

    return runner


# --- train sample
def train(timeout=10):
    runner = _create_runner(load_parameter=False)

    # sequence training
    runner.train(timeout=timeout)

    # save parameter
    runner.save_parameter(_parameter_path)


# --- evaluate sample
def evaluate():
    runner = _create_runner(load_parameter=True)
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


# --- render terminal sample
def render_terminal():
    runner = _create_runner(load_parameter=True)
    runner.render_terminal()


# --- render window sample
#  (Run "pip install pillow pygame" to use the render_window)
def render_window():
    runner = _create_runner(load_parameter=True)
    runner.render_window()


# --- animation sample
#  (Run "pip install opencv-python pillow pygame" to use the animation)
def animation():
    runner = _create_runner(load_parameter=True)
    runner.animation_save_gif("_FrozenLake.gif")


# --- replay window sample
#  (Run "pip install opencv-python pillow pygame" to use the replay_window)
def replay_window():
    runner = _create_runner(load_parameter=True)
    runner.replay_window()


if __name__ == "__main__":
    train()
    evaluate()
    render_terminal()
    render_window()
    animation()
    replay_window()
