import numpy as np

import srl
from srl.algorithms import ppo
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ppo.Config(
        batch_size=128,
        memory_warmup_size=1000,
        discount=0.9,
        gae_discount=0.9,
        optimizer_initial_lr=0.01,
        optimizer_final_lr=0.001,
        optimizer_lr_step=100_000,
        enable_value_clip=True,
        enable_state_normalized=False,
        baseline_type="normal",
        experience_collection_method="GAE",
        surrogate_type="clip",  # kl or clip
    )
    rl_config.memory.capacity = 1000
    rl_config.hidden_block.set_mlp((64, 64))
    rl_config.value_block.set_mlp((64,))
    rl_config.policy_block.set_mlp(())

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary(expand_nested=True)

    # --- train
    runner.train(max_train_count=30_000)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
