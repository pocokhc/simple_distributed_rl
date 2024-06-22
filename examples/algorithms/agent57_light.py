import numpy as np

import srl
from srl.algorithms import agent57_light
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")

    rl_config = agent57_light.Config(
        enable_rescale=True,
        enable_double_dqn=True,
        actor_num=4,
        enable_intrinsic_reward=True,
        input_ext_reward=False,
        input_int_reward=False,
        input_action=False,
    )
    rl_config.hidden_block.set_dueling_network((128, 128))
    rl_config.batch_size = 64
    rl_config.memory_capacity = 100_000
    rl_config.set_replay_memory()
    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # --- train
    runner.train(max_episodes=200)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
