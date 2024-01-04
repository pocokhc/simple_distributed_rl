import numpy as np
import srl
from srl.algorithms import ddpg


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = ddpg.Config()

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # --- train
    runner.set_progress_options(interval_limit=30)
    runner.train(max_train_count=200 * 50)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    runner.replay_window()


if __name__ == "__main__":
    main()
