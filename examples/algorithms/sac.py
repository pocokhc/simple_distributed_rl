import numpy as np

import srl
from srl.algorithms import sac
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = sac.Config(
        lr_policy=0.003,
        lr_q=0.003,
        lr_alpha=0.001,
    )
    rl_config.policy_hidden_block.set_mlp((64, 64, 64))
    rl_config.q_hidden_block.set_mlp((128, 128, 128))
    runner = srl.Runner(env_config, rl_config)
    runner.model_summary(expand_nested=True)

    # --- train
    runner.train(max_episodes=25)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"Average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
