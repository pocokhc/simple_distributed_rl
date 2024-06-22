import os

import numpy as np

import srl
from srl.algorithms import agent57, agent57_light, ddpg, dqn, dreamer_v3, ppo, r2d2, rainbow, sac
from srl.utils import common

base_dir = os.path.dirname(__file__)
ENV_NAME = "Pendulum-v1"

BASE_TRAIN = 200 * 200
BASE_LR = 0.001
BASE_BLOCK = (64, 64)


def _run(name, rl_config, is_mp):
    common.logger_print()

    rl_config.memory_compress = False
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_NAME}_{name}" + ("_mp" if is_mp else "")),
        enable_eval=True,
        eval_episode=10,
    )
    runner.model_summary()
    if is_mp:
        runner.train_mp(max_train_count=BASE_TRAIN)
    else:
        runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate()
    print(f"[{name}] evaluate episodes: {np.mean(rewards)}")


def main_dqn(is_mp):
    rl_config = dqn.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.hidden_block.set(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    _run("DQN", rl_config, is_mp)


def main_rainbow(is_mp):
    rl_config = rainbow.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    _run("Rainbow", rl_config, is_mp)


def main_agent57_light(is_mp):
    rl_config = agent57_light.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.enable_intrinsic_reward = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    _run("Agent57_light", rl_config, is_mp)


def main_r2d2(is_mp):
    rl_config = r2d2.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 1000
    _run("R2D2", rl_config, is_mp)


def main_agent57(is_mp):
    rl_config = agent57.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
    )
    rl_config.enable_intrinsic_reward = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 1000
    _run("Agent57", rl_config, is_mp)


def main_ppo(is_mp):
    rl_config = ppo.Config(lr=BASE_LR)
    rl_config.hidden_block.set(BASE_BLOCK)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.value_block.set(BASE_BLOCK)
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 1000
    _run("PPO", rl_config, is_mp)


def main_ddpg(is_mp):
    rl_config = ddpg.Config(lr=BASE_LR)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.q_block.set(BASE_BLOCK)
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 1000
    _run("DDPG", rl_config, is_mp)


def main_sac(is_mp):
    rl_config = sac.Config(lr_policy=BASE_LR, lr_q=BASE_LR)
    rl_config.policy_hidden_block.set(BASE_BLOCK)
    rl_config.q_hidden_block.set(BASE_BLOCK)
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 1000
    _run("SAC", rl_config, is_mp)


def main_dreamer_v3(is_mp):
    rl_config = dreamer_v3.Config(lr_model=BASE_LR, lr_critic=BASE_LR, lr_actor=BASE_LR)
    rl_config.set_dreamer_v3()
    rl_config.rssm_deter_size = 64
    rl_config.rssm_stoch_size = 4
    rl_config.rssm_classes = 4
    rl_config.rssm_hidden_units = 256
    rl_config.reward_layer_sizes = (256,)
    rl_config.cont_layer_sizes = (256,)
    rl_config.encoder_decoder_mlp = BASE_BLOCK
    rl_config.critic_layer_sizes = BASE_BLOCK
    rl_config.actor_layer_sizes = BASE_BLOCK
    rl_config.batch_size = 32
    rl_config.batch_length = 15
    rl_config.horizon = 5
    rl_config.memory_capacity = 10_000
    rl_config.memory_warmup_size = 50
    rl_config.free_nats = 0.1
    rl_config.warmup_world_model = 1_000
    _run("DreamerV3", rl_config, is_mp)


def compare1():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_NAME}_DQN"),
            os.path.join(base_dir, f"_{ENV_NAME}_Rainbow"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57_light"),
            os.path.join(base_dir, f"_{ENV_NAME}_R2D2"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57"),
            os.path.join(base_dir, f"_{ENV_NAME}_PPO"),
            os.path.join(base_dir, f"_{ENV_NAME}_DDPG"),
            os.path.join(base_dir, f"_{ENV_NAME}_SAC"),
            os.path.join(base_dir, f"_{ENV_NAME}_DreamerV3"),
        ]
    )
    histories.plot(
        "train",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )
    histories.plot(
        "time",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )


def compare2():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_NAME}_DQN_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Rainbow_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57_light_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_R2D2_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_PPO_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_DDPG_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_SAC_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_DreamerV3_mp"),
        ]
    )
    histories.plot(
        "train",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )
    histories.plot(
        "time",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )


if __name__ == "__main__":
    main_dqn(is_mp=False)
    main_dqn(is_mp=True)
    main_rainbow(is_mp=False)
    main_rainbow(is_mp=True)
    main_agent57_light(is_mp=False)
    main_agent57_light(is_mp=True)
    main_r2d2(is_mp=False)
    main_r2d2(is_mp=True)
    main_agent57(is_mp=False)
    main_agent57(is_mp=True)
    main_ppo(is_mp=False)
    main_ppo(is_mp=True)
    main_ddpg(is_mp=False)
    main_ddpg(is_mp=True)
    main_sac(is_mp=False)
    main_sac(is_mp=True)
    main_dreamer_v3(is_mp=False)
    main_dreamer_v3(is_mp=True)
    compare1()
    compare2()
