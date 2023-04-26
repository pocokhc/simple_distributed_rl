import os

import matplotlib.pyplot as plt

import srl
from srl import runner
from srl.rl.models import MLPBlockConfig

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.algorithms import dqn, rainbow  # isort: skip


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    rl_configs.append(
        (
            "DQN",
            dqn.Config(
                hidden_block_config=MLPBlockConfig(layer_sizes=(64, 64)),
                lr=0.0005,
                enable_double_dqn=False,
                enable_rescale=False,
            ),
        )
    )

    rainbow_base = dict(
        lr=0.0005,
        hidden_layer_sizes=(64, 64),
        enable_double_dqn=False,
        enable_dueling_network=False,
        enable_noisy_dense=False,
        multisteps=1,
        memory_name="ReplayMemory",
        enable_rescale=False,
    )
    rl_configs.append(("base", rainbow.Config(**rainbow_base)))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.enable_double_dqn = True
    rl_configs.append(("double_dqn", rl_config))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.enable_dueling_network = True
    rl_configs.append(("dueling_network", rl_config))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.enable_noisy_dense = True
    rl_configs.append(("noisy_dense", rl_config))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.multisteps = 10
    rl_configs.append(("multisteps 10", rl_config))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.memory_name = "ProportionalMemory"
    rl_config.memory_alpha = 1.0
    rl_config.memory_beta_initial = 1.0
    rl_configs.append(("ProportionalMemory", rl_config))

    rl_config = rainbow.Config(**rainbow_base)
    rl_config.enable_double_dqn = True
    rl_config.enable_dueling_network = True
    rl_config.enable_noisy_dense = True
    rl_config.multisteps = 10
    rl_config.memory_name = "ProportionalMemory"
    rl_config.memory_alpha = 1.0
    rl_config.memory_beta_initial = 0.8
    rl_config.memory_beta_steps = 200 * 50
    rl_configs.append(("Rainbow", rl_config))

    results = []
    for name, rl_config in rl_configs:
        config = runner.Config(env_config, rl_config)
        print(name)

        # --- train
        _, _, history = runner.train(config, max_episodes=200, enable_file_logger=True)
        results.append((name, history))

    plt.xlabel("episode")
    plt.ylabel("valid reward")
    for name, h in results:
        df = h.get_df()
        plt.plot(df["episode_count"], df["eval_reward0"].rolling(50).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_rainbow.png"))
    plt.show()


if __name__ == "__main__":
    main()
