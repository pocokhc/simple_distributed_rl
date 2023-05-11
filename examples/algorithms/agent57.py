import os

import matplotlib.pyplot as plt

import srl
from srl import runner
from srl.rl import memories
from srl.utils import common

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.algorithms import agent57  # isort: skip

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    base_config = dict(
        lstm_units=128,
        hidden_layer_sizes=(64, 64),
        enable_dueling_network=False,
        memory=memories.ReplayMemoryConfig(100_000),
        target_model_update_interval=100,
        enable_rescale=False,
        burnin=5,
        sequence_length=5,
        enable_retrace=False,
        enable_intrinsic_reward=False,
        actor_num=1,
        input_ext_reward=False,
        input_int_reward=False,
        input_action=False,
    )

    # base
    rl_configs.append(("base", agent57.Config(**base_config)))

    # intrinsic_reward
    rl_config = agent57.Config(**base_config)
    rl_config.enable_intrinsic_reward = True
    rl_configs.append(("intrinsic_reward", rl_config))

    # retrace
    rl_config = agent57.Config(**base_config)
    rl_config.enable_retrace = True
    rl_configs.append(("retrace", rl_config))

    # actor
    rl_config = agent57.Config(**base_config)
    rl_config.actor_num = 16
    rl_configs.append(("actor_16", rl_config))

    # UVFA
    rl_config = agent57.Config(**base_config)
    rl_config.input_ext_reward = True
    rl_config.input_int_reward = True
    rl_config.input_action = True
    rl_configs.append(("UVFA", rl_config))

    # train
    results = []
    for name, rl_config in rl_configs:
        print(name)
        config = runner.Config(env_config, rl_config)
        _, _, history = runner.train(config, max_episodes=200, enable_evaluation=False, enable_file_logger=True)
        results.append((name, history))

    # plot
    plt.figure(figsize=(12, 6))
    plt.xlabel("episode")
    plt.ylabel("reward")
    for name, h in results:
        df = h.get_df()
        plt.plot(df["episode_count"], df["episode_reward0"].rolling(10).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_agent57.png"))
    plt.show()


if __name__ == "__main__":
    main()
