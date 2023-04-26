import numpy as np

import srl
from srl import runner
from srl.utils import common

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.algorithms import agent57_light  # isort: skip

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")

    rl_config = agent57_light.Config(
        hidden_layer_sizes=(64, 64),
        enable_rescale=True,
        #
        enable_double_dqn=True,
        enable_dueling_network=True,
        memory_name="ReplayMemory",
        #
        actor_num=4,
        enable_intrinsic_reward=True,
        input_ext_reward=False,
        input_int_reward=False,
        input_action=False,
    )

    config = runner.Config(env_config, rl_config)
    config.model_summary()

    # --- train
    parameter, remote_memory, history = runner.train(config, max_episodes=200)

    # --- evaluate
    rewards = runner.evaluate(config, parameter)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
