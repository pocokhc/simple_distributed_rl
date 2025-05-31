import numpy as np

import srl
from srl.algorithms import sac
from srl.utils import common

common.logger_print()


def main_discrete():
    env_config = srl.EnvConfig("Grid")
    rl_config = sac.Config()
    rl_config.batch_size = 32
    rl_config.lr_policy = 0.0002
    rl_config.lr_q = 0.001
    rl_config.memory.capacity = 10000
    rl_config.memory.warmup_size = 1000
    rl_config.policy_hidden_block.set((64, 64))
    rl_config.q_hidden_block.set((64, 64))
    rl_config.entropy_bonus_exclude_q = True
    rl_config.entropy_alpha = 0.1
    rl_config.entropy_alpha_auto_scale = False

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()
    runner.set_progress(interval_limit=30)

    # --- train
    runner.train(max_train_count=5000)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"Average reward: {np.mean(rewards)}")

    runner.replay_window()


def main_continuous():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = sac.Config()
    rl_config.batch_size = 32
    rl_config.lr_policy = 0.003
    rl_config.lr_q = 0.003
    rl_config.memory.capacity = 10000
    rl_config.memory.warmup_size = 1000
    rl_config.start_steps = 1000
    rl_config.policy_hidden_block.set((64, 64, 64))
    rl_config.q_hidden_block.set((128, 128, 128))
    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # --- train
    runner.train(max_episodes=50)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"Average reward: {np.mean(rewards)}")

    runner.replay_window()


if __name__ == "__main__":
    # main_discrete()
    main_continuous()
