import numpy as np

import srl
from srl import runner
from srl.utils import common

# --- algorithm load
from srl.algorithms import dqn  # isort: skip


common.set_logger()


def main():

    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = dqn.Config(
        hidden_block_kwargs=dict(layer_sizes=(64, 64)),
        capacity=10_000,
        lr=0.001,
        epsilon=0.1,
        memory_warmup_size=1000,
        target_model_update_interval=1000,
    )
    config = runner.Config(env_config, rl_config)

    # --- train
    if True:
        parameter, remote_memory, history = runner.train(
            config,
            max_train_count=50_000,
            enable_evaluation=False,
        )
    else:
        parameter, remote_memory, history = runner.mp_train(
            config,
            max_train_count=50_000,
            enable_evaluation=False,
        )

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10)
    print(f"reward 10 episode mean: {np.mean(rewards)}")

    # --- animation
    render = runner.animation(config, parameter)
    render.create_anime(draw_info=True).save("_Pendulum.gif")


if __name__ == "__main__":
    main()
