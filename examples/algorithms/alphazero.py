import os

import numpy as np

import srl
from srl import runner
from srl.rl.models import alphazero as alphazero_model
from srl.rl.models import mlp
from srl.utils import common

# --- env & algorithm load
from srl.envs import othello  # isort: skip # noqa F401
from srl.algorithms import alphazero  # isort: skip

common.logger_print()


def main():
    rl_config = alphazero.Config(
        input_image_block=alphazero_model.AlphaZeroImageBlockConfig(n_blocks=9, filters=64),
        value_block=mlp.MLPBlockConfig((128,)),
        policy_block=mlp.MLPBlockConfig((128,)),
        num_simulations=100,
        sampling_steps=1,
        lr_schedule=[
            {"train": 0, "lr": 0.001},
            {"train": 1000, "lr": 0.0005},
            {"train": 5000, "lr": 0.0002},
        ],
        batch_size=128,
        memory_warmup_size=500,
        capacity=100_000,
    )

    env_config = srl.EnvConfig("Othello4x4")

    """
    othello.LayerProcessor を適用する事で入力形式を変えています。
    変更後は WxHx2 の画像レイヤーで、WxHが盤面に相当します。
    """
    rl_config.processors = [othello.LayerProcessor()]

    config = runner.Config(env_config, rl_config)

    # --- train
    config.players = [None, None]
    parameter, _, _ = runner.train(
        config,
        max_episodes=2000,
        enable_evaluation=False,
    )

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
        ["random", "cpu"],
        ["cpu", "random"],
    ]:
        config.players = players
        rewards = runner.evaluate(config, parameter, max_episodes=100)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")

    # --- rendering
    config.players = [None, "cpu"]
    render = runner.animation(config, parameter)
    render.create_anime(draw_info=True).save(os.path.join(os.path.dirname(__file__), "_alphazero.gif"))

    # --- 対戦
    config.players = [None, "human"]
    runner.render(config, parameter)


if __name__ == "__main__":
    main()
