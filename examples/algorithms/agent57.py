import os

import matplotlib.pyplot as plt

import srl
from srl.algorithms import agent57
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    base_config = agent57.Config(
        lstm_units=128,
        batch_size=64,
        lr_ext=0.001,
        lr_int=0.001,
        target_model_update_interval=100,
        enable_rescale=False,
        burnin=5,
        sequence_length=5,
        enable_intrinsic_reward=False,
        actor_num=1,
        input_ext_reward=False,
        input_int_reward=False,
        input_action=False,
    )
    base_config.hidden_block.set((64, 64))
    base_config.memory.capacity = 100_000

    # base
    rl_configs.append(("base", base_config.copy()))

    # intrinsic_reward
    _c: agent57.Config = base_config.copy()
    _c.enable_intrinsic_reward = True
    rl_configs.append(("intrinsic_reward", _c))

    # actor
    _c = base_config.copy()
    _c.actor_num = 16
    rl_configs.append(("actor_16", _c))

    # UVFA
    _c = base_config.copy()
    _c.input_ext_reward = True
    _c.input_int_reward = True
    _c.input_action = True
    rl_configs.append(("UVFA", _c))

    # train
    results = []
    for name, rl_config in rl_configs:
        print(name)
        runner = srl.Runner(env_config, rl_config)
        runner.set_history_on_memory()
        runner.train(max_train_count=200 * 100)
        df = runner.get_history().get_df()
        results.append((name, df))

    # plot
    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel("reward")
    for name, df in results:
        df = df[df["name"] == "actor0"]
        plt.plot(df["train"], df["reward0"].rolling(10).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_agent57.png"))
    plt.show()


if __name__ == "__main__":
    main()
