import numpy as np
import srl
from srl.algorithms import ppo
from srl.base.define import SpaceTypes
from srl.utils import common

common.logger_print()


def main(mode: str):
    env_config = srl.EnvConfig("Grid")
    rl_config = ppo.Config(
        batch_size=32,
        baseline_type="",
        experience_collection_method="GAE",
        surrogate_type="clip",  # kl or clip
    )
    rl_config.lr = 0.005
    rl_config.memory_capacity = 1000
    rl_config.memory_warmup_size = 1000
    rl_config.hidden_block.set((64, 64))
    rl_config.value_block.set(())
    rl_config.policy_block.set(())

    if mode == "CONTINUOUS":
        TRAIN_COUNT = 50000
        rl_config.lr = 0.0005
        rl_config.entropy_weight = 1.0
        rl_config.override_action_type = SpaceTypes.CONTINUOUS
    else:
        TRAIN_COUNT = 20000

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # --- train
    runner.train(max_train_count=TRAIN_COUNT)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    runner.replay_window()


if __name__ == "__main__":
    main("DISCRETE")
    # main("CONTINUOUS")
