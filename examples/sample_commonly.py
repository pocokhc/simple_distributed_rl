import numpy as np

import srl
from srl import runner

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip

# --- save parameter path
_parameter_path = "_parameter_ox_QL.dat"


def _create_config(load_parameter: bool):
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)
    parameter = config.make_parameter()

    if load_parameter:
        parameter.load(_parameter_path)

    return config, parameter


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


def evaluate():
    config, parameter = _create_config(load_parameter=True)
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


def render():
    config, parameter = _create_config(load_parameter=True)
    runner.render(config, parameter)


def render_window():
    config, parameter = _create_config(load_parameter=True)
    runner.render_window(config, parameter)


def animation():
    config, parameter = _create_config(load_parameter=True)
    render = runner.animation(config, parameter)
    render.create_anime().save("_OX.gif")


if __name__ == "__main__":
    train()
    evaluate()
    render()
    render_window()
    animation()
