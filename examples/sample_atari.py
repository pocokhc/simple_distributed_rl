import os
import warnings

import numpy as np
import srl
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.runner import mp, sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore")


def main(is_mp):
    env_config = srl.envs.Config("ALE/Galaxian-v5")
    rl_config = srl.rl.rainbow.Config(window_length=4, multisteps=10)

    # atari processor
    rl_config.processors = [ImageProcessor(gray=True, resize=(84, 84), enable_norm=True)]

    config = sequence.Config(env_config, rl_config)

    # atari play config
    config.skip_frames = 4
    config.max_episode_steps = 500

    # (option) print tensorflow model
    config.model_summary()

    # load parameter
    # config.set_parameter_path(parameter_path="tmp/Rainbow_params.dat")

    # --- train
    if not is_mp:
        # sequence training
        parameter, remote_memory, history = sequence.train(config, timeout=60 * 30)
    else:
        # distribute training
        mp_config = mp.Config(worker_num=1)
        parameter, remote_memory = mp.train(config, mp_config, timeout=60 * 30)

    # save parameter
    # parameter.save("tmp/Rainbow_params.dat")

    # --- test
    rewards = sequence.evaluate(config, parameter, max_episodes=5)
    print(f"reward: {np.mean(rewards)}")

    # --- rendering
    _, render = sequence.render(config, parameter, mode="", enable_animation=True)

    # save animation
    render.create_anime().save("tmp/Galaxian.gif")


if __name__ == "__main__":
    main(is_mp=False)
    # main(is_mp=True)
