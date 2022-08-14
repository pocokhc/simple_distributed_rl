import warnings

import numpy as np
import srl
from srl.runner import mp, sequence

warnings.simplefilter("ignore")


def main(is_mp):
    env_config = srl.envs.Config("ALE/Breakout-v5")

    rl_config = srl.rl.rainbow.Config(multisteps=5, memory_beta_initial=0.8, lr=0.0001)
    rl_config.window_length = 4

    config = sequence.Config(env_config, rl_config)

    # atari play config
    config.skip_frames = 8
    config.max_episode_steps = 500

    # load parameter
    # rl_config.parameter_path = "tmp/Rainbow_params.dat"

    # (option) print tensorflow model
    config.model_summary()

    # --- train
    if not is_mp:
        # sequence training
        parameter, remote_memory, history = sequence.train(config, timeout=60 * 60 * 4)
    else:
        # distribute training
        mp_config = mp.Config(actor_num=1)
        parameter, remote_memory, history = mp.train(config, mp_config, timeout=60 * 60 * 4)

    # save parameter
    # parameter.save("tmp/Rainbow_params.dat")

    # --- test
    rewards = sequence.evaluate(config, parameter, max_episodes=5)
    print(f"reward: {np.mean(rewards)}")

    # --- rendering
    _, render = sequence.render(config, parameter, render_terminal=False, enable_animation=True)

    # save animation
    render.create_anime(interval=1000 / 60).save("_Breakout.gif")


if __name__ == "__main__":
    main(is_mp=False)
    # main(is_mp=True)
