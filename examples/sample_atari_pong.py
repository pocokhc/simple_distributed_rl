from srl import rl
from srl.base.define import EnvObservationType
from srl.rl.processor import ImageProcessor
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, Rendering, RenderingAnimation
from srl.runner.callbacks_mp import TrainFileLogger


def main_use_runner(is_mp):
    config = sequence.Config(
        env_name="ALE/Pong-v5",
        rl_config=rl.rainbow.Config(window_length=4),
    )

    # atari config
    config.skip_frames = 4
    config.max_episode_steps = 100
    config.override_env_observation_type = EnvObservationType.COLOR
    config.processors = [ImageProcessor(gray=True, resize=(84, 84), enable_norm=True)]

    # (option) print tensorflow model
    config.create_parameter().summary()

    # --- train
    if not is_mp:
        # sequence training
        config.set_play_config(timeout=10, training=True, callbacks=[PrintProgress()])
        episode_rewards, parameter, memory = sequence.play(config)
    else:
        # distibute training
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(
            timeout=60 * 60, callbacks=[TrainFileLogger(enable_log=True, enable_checkpoint=False)]
        )
        parameter = mp.train(config, mp_config)

    # --- test
    config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
    sequence.play(config, parameter)

    # --- rendering
    render = RenderingAnimation()
    config.set_play_config(max_episodes=1, callbacks=[Rendering(), render])
    sequence.play(config, parameter)

    # save animation
    render.create_anime().save("tmp/Pong.gif")


if __name__ == "__main__":

    main_use_runner(is_mp=False)
    # main_use_runner(is_mp=True)
