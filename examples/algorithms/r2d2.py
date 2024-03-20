import os

import matplotlib.pyplot as plt

import srl
from srl.algorithms import r2d2


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    # ハイパーパラメータ
    r2d2_base = r2d2.Config(
        lstm_units=64,
        target_model_update_interval=100,
        enable_rescale=False,
        burnin=5,
        sequence_length=5,
        enable_retrace=False,
    )
    r2d2_base.hidden_block.set((64,))
    r2d2_base.memory.capacity = 100_000

    # no retrace
    rl_configs.append(("no retrace", r2d2_base.copy()))

    # retrace
    _c = r2d2_base.copy()
    _c.enable_retrace = True
    rl_configs.append(("retrace", _c))

    # train
    results = []
    for name, rl_config in rl_configs:
        print(name)
        runner = srl.Runner(env_config, rl_config)
        runner.set_history_on_memory(enable_eval=True)
        runner.train(max_episodes=50)
        results.append((name, runner.get_history().get_df()))

    # plot
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("reward")
    for name, df in results:
        plt.plot(df["episode"], df["eval_reward0"].rolling(10).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_r2d2.png"))
    plt.show()


if __name__ == "__main__":
    main()
