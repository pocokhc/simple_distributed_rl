import numpy as np
import srl
from srl import runner


def main():

    env_config = srl.envs.Config("Grid")
    rl_config = srl.rl.ql.Config()
    config = runner.Config(env_config, rl_config)

    # load parameter
    # rl_config.parameter_path = "tmp/QL_params.dat"

    # --- training
    if True:
        # sequence training
        parameter, memory, history = runner.train(config, timeout=10)
    else:
        # distributed training
        mp_config = runner.MpConfig(actor_num=2)
        parameter, memory, history = runner.mp_train(config, mp_config, timeout=10)

    # save parameter
    # parameter.save("tmp/QL_params.dat")

    # --- test
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"test reward mean: {np.mean(rewards)}")

    # --- rendering
    runner.render(config, parameter)


if __name__ == "__main__":
    main()
