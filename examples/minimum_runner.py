import os

import numpy as np
import srl
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, Rendering
from srl.runner.callbacks_mp import TrainFileLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(is_mp):

    env_config = srl.envs.Config("FrozenLake-v1")
    rl_config = srl.rl.ql.Config()
    config = sequence.Config(env_config, rl_config)

    # load parameter
    # config.set_parameter_path(parameter_path="tmp/QL_params.dat")

    # --- training
    if not is_mp:
        # sequence training
        config.set_train_config(timeout=30 * 1, callbacks=[PrintProgress()])
        parameter, memory = sequence.train(config)
    else:
        # distributed training
        mp_config = mp.Config(worker_num=2)
        config.set_train_config()
        mp_config.set_train_config(
            timeout=30 * 1, callbacks=[TrainFileLogger(enable_log=False, enable_checkpoint=False)]
        )
        parameter, memory = mp.train(config, mp_config)

    # save parameter
    # parameter.save("tmp/QL_params.dat")

    # --- test
    config.set_play_config(max_episodes=100)
    rewards, _, _ = sequence.play(config, parameter)
    print(f"test reward mean: {np.mean(rewards)}")

    # --- rendering
    render = Rendering(step_stop=False)
    config.set_play_config(max_episodes=1, callbacks=[render])
    sequence.play(config, parameter)


if __name__ == "__main__":
    main(is_mp=False)
    # main(is_mp=True)
