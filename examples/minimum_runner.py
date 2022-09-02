import os
import sys

import numpy as np
import srl
from srl import runner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

# --- env & algorithm
from envs import grid  # isort: skip # noqa F401
from algorithms import ql  # isort: skip


def main():

    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)

    # load parameter
    # rl_config.parameter_path = "tmp/QL_params.dat"

    # --- training
    if True:
        # sequence training
        parameter, memory, history = runner.train(config, timeout=10)
    else:
        # distributed training
        mp_config = runner.MpConfig(actor_num=2)
        parameter, memory, history = runner.mp_train(config, mp_config, timeout=10)

    # save parameter
    # parameter.save("tmp/QL_params.dat")

    # --- test
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"test reward mean: {np.mean(rewards)}")

    # --- rendering
    runner.render(config, parameter)


if __name__ == "__main__":
    main()
