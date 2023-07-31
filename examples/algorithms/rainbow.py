import os

import matplotlib.pyplot as plt

import srl
from srl.algorithms import dqn, rainbow


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    _c = dqn.Config(
        lr=0.0005,
        enable_double_dqn=False,
        enable_rescale=False,
    )
    _c.hidden_block.set_mlp((64, 64))
    rl_configs.append(("DQN", _c))

    rainbow_base_config = rainbow.Config(
        lr=0.0005,
        hidden_layer_sizes=(64, 64),
        enable_double_dqn=False,
        enable_dueling_network=False,
        enable_noisy_dense=False,
        multisteps=1,
        enable_rescale=False,
    )
    rainbow_base_config.memory.set_replay_memory()

    rl_configs.append(("base", rainbow_base_config.copy()))

    _c = rainbow_base_config.copy()
    _c.enable_double_dqn = True
    rl_configs.append(("double_dqn", _c))

    _c = rainbow_base_config.copy()
    _c.enable_dueling_network = True
    rl_configs.append(("dueling_network", _c))

    _c = rainbow_base_config.copy()
    _c.enable_noisy_dense = True
    rl_configs.append(("noisy_dense", _c))

    _c = rainbow_base_config.copy()
    _c.multisteps = 10
    rl_configs.append(("multisteps 10", _c))

    _c = rainbow_base_config.copy()
    _c.memory.set_proportional_memory(alpha=1.0, beta_initial=1.0)
    rl_configs.append(("ProportionalMemory", _c))

    _c = rainbow_base_config.copy()
    _c.enable_double_dqn = True
    _c.enable_dueling_network = True
    _c.enable_noisy_dense = True
    _c.multisteps = 10
    _c.memory.set_proportional_memory(alpha=1.0, beta_initial=0.8, beta_steps=200 * 50)
    rl_configs.append(("Rainbow", _c))

    results = []
    for name, rl_config in rl_configs:
        runner = srl.Runner(env_config, rl_config)
        print(name)

        # --- train
        runner.set_history()
        runner.train(max_episodes=200)
        results.append((name, runner.get_history().get_df()))

    plt.xlabel("episode")
    plt.ylabel("valid reward")
    for name, df in results:
        plt.plot(df["episode"], df["eval_reward0"].rolling(50).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_rainbow.png"))
    plt.show()


if __name__ == "__main__":
    main()
