import os

import srl
from srl.algorithms import dreamer_v3
from srl.utils import common


def train_grid(mode: str):
    env_config = srl.EnvConfig("Grid")
    rl_config = dreamer_v3.Config()
    if mode == "v1":
        rl_config.set_dreamer_v1()
    elif mode == "v2":
        rl_config.set_dreamer_v2()
    elif mode == "v3":
        rl_config.set_dreamer_v3()

    # model
    rl_config.rssm_deter_size = 8
    rl_config.rssm_stoch_size = 16
    rl_config.rssm_classes = 16
    rl_config.rssm_hidden_units = 32
    rl_config.reward_layer_sizes = (32,)
    rl_config.cont_layer_sizes = (32,)
    rl_config.critic_layer_sizes = (128,)
    rl_config.actor_layer_sizes = (128,)
    rl_config.encoder_decoder_mlp = (16, 16)
    # lr
    rl_config.batch_size = 32
    rl_config.batch_length = 5
    rl_config.lr_model = 0.0005
    rl_config.lr_critic = 0.0003
    rl_config.lr_actor = 0.0001
    rl_config.horizon = 3
    # memory
    rl_config.memory.warmup_size = 50
    rl_config.memory.capacity = 10_000

    rl_config.encoder_decoder_dist = "linear"
    rl_config.free_nats = 0.01
    rl_config.warmup_world_model = 1_000

    # --- train
    runner = srl.Runner(env_config, rl_config)
    runner.summary()

    if mode == "v1":
        runner.train(max_train_count=3_000)
    elif mode == "v2":
        runner.train(max_train_count=3_000)
    elif mode == "v3":
        rl_config.warmup_world_model = 2_000
        runner.train(max_train_count=4_000)

    path = os.path.join(os.path.dirname(__file__), f"_dreamer_{mode}.gif")
    runner.animation_save_gif(path)
    runner.replay_window()


def train_Pendulum():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = dreamer_v3.Config()
    rl_config.set_dreamer_v3()

    # model
    rl_config.rssm_deter_size = 64
    rl_config.rssm_stoch_size = 4
    rl_config.rssm_classes = 4
    rl_config.rssm_hidden_units = 512
    rl_config.reward_layer_sizes = (256,)
    rl_config.cont_layer_sizes = (256,)
    rl_config.critic_layer_sizes = (128, 128)
    rl_config.actor_layer_sizes = (128, 128)
    rl_config.encoder_decoder_mlp = (64, 64)
    # lr
    rl_config.batch_size = 32
    rl_config.batch_length = 15
    rl_config.lr_model = 0.0001
    rl_config.lr_critic = 0.0001
    rl_config.lr_actor = 0.00002
    rl_config.horizon = 5
    # memory
    rl_config.memory.warmup_size = 50
    rl_config.memory.capacity = 10_000

    rl_config.encoder_decoder_dist = "linear"
    rl_config.free_nats = 0.1
    rl_config.warmup_world_model = 1_000

    # --- train
    runner = srl.Runner(env_config, rl_config)
    runner.train(max_train_count=30_000)

    path = os.path.join(os.path.dirname(__file__), "_dreamer_Pendulum.gif")
    runner.animation_save_gif(path)
    # runner.replay_window()


if __name__ == "__main__":
    common.logger_print()

    train_grid("v1")
    # train_grid("v2")
    # train_grid("v3")

    # train_Pendulum()
