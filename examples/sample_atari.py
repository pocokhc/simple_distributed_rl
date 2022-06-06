import os
import warnings

import numpy as np
import srl
from srl.base.define import EnvObservationType, RenderType
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, Rendering
from srl.runner.callbacks_mp import TrainFileLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore")


def main(is_mp):
    env_config = srl.envs.Config("ALE/Galaxian-v5")
    rl_config = srl.rl.rainbow.Config(window_length=4, multisteps=10)

    # atari config
    rl_config.override_env_observation_type = EnvObservationType.COLOR
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
        config.set_train_config(timeout=60 * 30, callbacks=[PrintProgress()])
        parameter, remote_memory = sequence.train(config)
    else:
        # distribute training
        mp_config = mp.Config(worker_num=1)
        config.set_train_config()
        mp_config.set_train_config(
            timeout=60 * 30, callbacks=[TrainFileLogger(enable_log=True, enable_checkpoint=False)]
        )
        parameter, remote_memory = mp.train(config, mp_config)

    # save parameter
    # parameter.save("tmp/Rainbow_params.dat")

    # --- test
    config.set_play_config(max_episodes=5)
    rewards, _, _ = sequence.play(config, parameter)
    print(f"reward: {np.mean(rewards)}")

    # --- rendering
    render = Rendering(mode=RenderType.NONE, enable_animation=True)
    config.set_play_config(max_episodes=1, callbacks=[render])
    sequence.play(config, parameter)

    # save animation
    render.create_anime().save("tmp/Galaxian.gif")


if __name__ == "__main__":
    main(is_mp=False)
    # main(is_mp=True)
