import os

import matplotlib.pyplot as plt

import srl
from srl import runner
from srl.algorithms import r2d2

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.rl import memories  # isort: skip


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_configs = []

    # ハイパーパラメータ
    r2d2_base = dict(
        lstm_units=64,
        hidden_layer_sizes=(64,),
        enable_dueling_network=False,
        memory=memories.ReplayMemoryConfig(100_000),
        target_model_update_interval=100,
        enable_rescale=False,
        burnin=5,
        sequence_length=5,
        enable_retrace=False,
    )

    # no retrace
    rl_configs.append(("no retrace", r2d2.Config(**r2d2_base)))

    # retrace
    rl_config = r2d2.Config(**r2d2_base)
    rl_config.enable_retrace = True
    rl_configs.append(("retrace", rl_config))

    # train
    results = []
    for name, rl_config in rl_configs:
        print(name)
        config = runner.Config(env_config, rl_config)
        _, _, history = runner.train(
            config,
            max_episodes=50,
            history=runner.HistoryOption(),
            eval=runner.EvalOption(),
        )
        results.append((name, history))

    # plot
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("reward")
    for name, h in results:
        df = h.get_df()
        plt.plot(df["actor0_episode"], df["actor0_eval_reward0"].rolling(10).mean(), label=name)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_r2d2.png"))
    plt.show()


if __name__ == "__main__":
    main()
