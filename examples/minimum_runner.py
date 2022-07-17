import numpy as np
import srl
from srl.runner import mp, sequence


def main(is_mp):

    env_config = srl.envs.Config("Grid")
    rl_config = srl.rl.ql.Config()
    config = sequence.Config(env_config, rl_config)

    # load parameter
    # rl_config.parameter_path = "tmp/QL_params.dat"

    # --- training
    if not is_mp:
        # sequence training
        parameter, memory, history = sequence.train(config, timeout=10)
    else:
        # distributed training
        mp_config = mp.Config(actor_num=2)
        parameter, memory, history = mp.train(config, mp_config, timeout=10)

    # save parameter
    # parameter.save("tmp/QL_params.dat")

    # --- test
    rewards = sequence.evaluate(config, parameter, max_episodes=100)
    print(f"test reward mean: {np.mean(rewards)}")

    # --- rendering
    sequence.render(config, parameter)


if __name__ == "__main__":
    main(is_mp=False)
    # main(is_mp=True)
