import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()

ENV_NAME = "Pendulum-v1"
BASE_LR = 0.001
BASE_TRAIN = 200 * 100
BASE_BLOCK = (64, 64)


def _run(rl_config, is_mp, is_image, train):
    if is_image:
        rl_config.observation_mode = "render_image"
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.summary()

    runner.set_mlflow()
    if is_mp:
        runner.train_mp(max_train_count=train)
    else:
        runner.train(max_train_count=train)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {np.mean(rewards)}, {rewards}")


def main_dqn(is_mp=False, is_image=False):
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        lr=BASE_LR,
        target_model_update_interval=2000,
        memory=dqn.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.hidden_block.set(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_rainbow(is_mp=False, is_image=False):
    from srl.algorithms import rainbow

    rl_config = rainbow.Config(
        lr=BASE_LR,
        target_model_update_interval=2000,
    )
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    rl_config.memory.compress = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_r2d2(is_mp=False, is_image=False):
    from srl.algorithms import r2d2

    rl_config = r2d2.Config(
        lr=BASE_LR,
        target_model_update_interval=2000,
        burnin=5,
        sequence_length=2,
    )
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    rl_config.memory.compress = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_agent57(is_mp=False, is_image=False):
    from srl.algorithms import agent57

    rl_config = agent57.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=2000,
        burnin=5,
        sequence_length=2,
        enable_intrinsic_reward=False,
    )
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    rl_config.memory.compress = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_ppo(is_mp=False, is_image=False):
    from srl.algorithms import ppo

    rl_config = ppo.Config(
        lr=BASE_LR,
        memory=ppo.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.hidden_block.set((128,))
    rl_config.policy_block.set((128,))
    rl_config.value_block.set((128,))
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_ddpg(is_mp=False, is_image=False):
    from srl.algorithms import ddpg

    rl_config = ddpg.Config(
        lr=BASE_LR,
        memory=ddpg.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.q_block.set(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_sac(is_mp=False, is_image=False):
    from srl.algorithms import sac

    rl_config = sac.Config(
        lr_policy=BASE_LR,
        lr_q=BASE_LR,
        memory=sac.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.policy_hidden_block.set(BASE_BLOCK)
    rl_config.q_hidden_block.set(BASE_BLOCK)
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def main_dreamer_v3(is_mp=False, is_image=False):
    from srl.algorithms import dreamer_v3

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
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 50
    rl_config.free_nats = 0.1
    rl_config.warmup_world_model = 1_000
    _run(rl_config, is_mp, is_image, BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "DQN",
        "Rainbow",
        "R2D2",
        "Agent57",
        "PPO",
        "DDPG",
        "SAC",
        "DreamerV3",
    ]:
        history = MLFlowCallback.get_metric(ENV_NAME, name, metric_name)
        if history is None:
            continue
        times = np.array([h.timestamp for h in history])
        times -= times[0]
        steps = [h.step for h in history]
        vals = [h.value for h in history]
        plt.plot(steps, common.rolling(vals), label=name)
    plt.grid()
    plt.legend()
    plt.title(f"Train:{BASE_TRAIN}, lr={BASE_LR}, Block={BASE_BLOCK}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_pendulum.png"))
    plt.show()


if __name__ == "__main__":
    is_mp = False
    is_image = False

    main_dqn(is_mp, is_image)
    main_rainbow(is_mp, is_image)
    main_r2d2(is_mp, is_image)
    main_agent57(is_mp, is_image)
    main_ppo(is_mp, is_image)
    main_ddpg(is_mp, is_image)
    main_sac(is_mp, is_image)
    main_dreamer_v3(is_mp, is_image)
    compare()
