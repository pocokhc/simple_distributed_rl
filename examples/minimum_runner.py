import srl
from srl import runner

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():

    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)

    # --- training
    if True:
        # sequence training
        parameter, memory, history = runner.train(config, timeout=10)
    else:
        # distributed training
        mp_config = runner.MpConfig(actor_num=2)
        parameter, memory, history = runner.mp_train(config, mp_config, timeout=10)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10)
    print(f"evaluate rewards: {rewards}")


if __name__ == "__main__":
    main()
