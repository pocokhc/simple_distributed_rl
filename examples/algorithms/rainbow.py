import os

import matplotlib.pyplot as plt
import numpy as np

import srl
from srl.algorithms import dqn, rainbow


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = rainbow.Config(
        lr=0.0005,
        enable_double_dqn=True,
        enable_noisy_dense=False,
        multisteps=1,
        enable_rescale=False,
    )
    rl_config.hidden_block.set_dueling_network((64, 64))
    rl_config.set_proportional_memory()
    runner = srl.Runner(env_config, rl_config)
    runner.set_device("CPU")
    runner.model_summary()

    # --- train
    runner.set_history_on_memory()
    runner.train(max_episodes=200)
    history = runner.get_history()
    history.plot()

    # --- evaluate
    rewards = runner.evaluate(max_episodes=20)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")


def main_compare():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    _c_dqn = dqn.Config(
        lr=0.0005,
        enable_double_dqn=False,
        enable_rescale=False,
    )
    _c_dqn.hidden_block.set((64, 64))
    rl_configs.append(("DQN", _c_dqn))

    rainbow_base_config = rainbow.Config(
        lr=0.0005,
        enable_double_dqn=False,
        enable_noisy_dense=False,
        multisteps=1,
        enable_rescale=False,
    )
    rainbow_base_config.hidden_block.set((64, 64))
    rainbow_base_config.set_replay_memory()

    rl_configs.append(("base", rainbow_base_config.copy()))

    _c: rainbow.Config = rainbow_base_config.copy()
    _c.enable_double_dqn = True
    rl_configs.append(("double_dqn", _c))

    _c = rainbow_base_config.copy()
    _c.hidden_block.set_dueling_network((64, 64))
    rl_configs.append(("dueling_network", _c))

    _c = rainbow_base_config.copy()
    _c.enable_noisy_dense = True
    rl_configs.append(("noisy_dense", _c))

    _c = rainbow_base_config.copy()
    _c.multisteps = 10
    rl_configs.append(("multisteps 10", _c))

    _c = rainbow_base_config.copy()
    _c.set_proportional_memory(alpha=1.0, beta_initial=1.0)
    rl_configs.append(("ProportionalMemory", _c))

    _c = rainbow_base_config.copy()
    _c.enable_double_dqn = True
    _c.hidden_block.set_dueling_network((64, 64))
    _c.enable_noisy_dense = True
    _c.multisteps = 10
    _c.set_proportional_memory(alpha=1.0, beta_initial=0.8, beta_steps=200 * 50)
    rl_configs.append(("Rainbow", _c))

    results = []
    for name, rl_config in rl_configs:
        runner = srl.Runner(env_config, rl_config)
        runner.set_device("CPU")
        print(name)

        # --- train
        runner.set_history_on_memory(enable_eval=True, interval=200, interval_mode="step")
        runner.train(max_episodes=200)
        results.append((name, runner.get_history().get_df()))

    plt.xlabel("step")
    plt.ylabel("valid reward")
    for name, df in results:
        plt.plot(df["step"], df["eval_reward0"].rolling(50).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_rainbow.png"))
    plt.show()


if __name__ == "__main__":
    main()
    # main_compare()
