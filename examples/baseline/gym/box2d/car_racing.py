import os

import mlflow
import numpy as np

import srl
from srl.runner.callbacks.mlflow_callback import MLFlowCallback
from srl.utils import common

mlflow.set_tracking_uri("mlruns")
common.logger_print()

ENV_NAME = "CarRacing-v2"
BASE_LR = 0.0005
BASE_TRAIN = 100_000
BASE_BLOCK = (512,)


def _run(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.model_summary()
    runner.train(max_train_count=train, callbacks=[MLFlowCallback()])
    rewards = runner.evaluate()
    print(f"[{rl_config.name}] evaluate episodes: {np.mean(rewards)}")


def main_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
    )
    rl_config.hidden_block.set(BASE_BLOCK)
    _run(rl_config, BASE_TRAIN)


def main_rainbow():
    from srl.algorithms import rainbow

    rl_config = rainbow.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, BASE_TRAIN)


def main_r2d2():
    from srl.algorithms import r2d2

    rl_config = r2d2.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, BASE_TRAIN)


def main_agent57():
    from srl.algorithms import agent57

    rl_config = agent57.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
        enable_intrinsic_reward=False,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    _run(rl_config, BASE_TRAIN)


def main_ppo():
    from srl.algorithms import ppo

    rl_config = ppo.Config(lr=BASE_LR)
    rl_config.hidden_block.set(BASE_BLOCK)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.value_block.set(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    rl_config.memory_compress = False
    _run(rl_config, BASE_TRAIN)


def main_ddpg():
    from srl.algorithms import ddpg

    rl_config = ddpg.Config(lr=BASE_LR)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.q_block.set(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    rl_config.memory_compress = False
    _run(rl_config, BASE_TRAIN)


def main_sac():
    from srl.algorithms import sac

    rl_config = sac.Config(lr_policy=BASE_LR, lr_q=BASE_LR)
    rl_config.policy_hidden_block.set(BASE_BLOCK)
    rl_config.q_hidden_block.set(BASE_BLOCK)
    rl_config.memory_warmup_size = 1000
    rl_config.memory_capacity = 10_000
    rl_config.memory_compress = False
    _run(rl_config, BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "DQN:tensorflow",
        "Rainbow:tensorflow",
        "R2D2",
        "Agent57:tensorflow",
        "PPO",
        "DDPG",
        "SAC",
    ]:
        steps, values = MLFlowCallback.get_metrics(ENV_NAME, name, metric_name)
        if steps is None:
            continue
        if len(values) > 20:
            values = common.moving_average(values, 10)
        plt.plot(steps, values, label=name)
    plt.grid()
    plt.legend()
    plt.title(f"Train:{BASE_TRAIN}, lr={BASE_LR}, Block={BASE_BLOCK}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_car_racing.png"))
    plt.show()


if __name__ == "__main__":
    main_dqn()
    main_rainbow()
    main_r2d2()
    main_agent57()
    main_ppo()
    main_ddpg()
    main_sac()
    compare()
