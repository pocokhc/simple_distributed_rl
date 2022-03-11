from srl import rl
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, RenderingEpisode
from srl.runner.callbacks_mp import TrainFileLogger


def main_use_runner(is_mp):
    config = sequence.Config(
        env_name="FrozenLake-v1",
        rl_config=rl.ql.Config(),
    )

    # --- train
    if not is_mp:
        # sequence training
        config.set_play_config(timeout=60, training=True, callbacks=[PrintProgress()])
        episode_rewards, parameter, memory = sequence.play(config)
    else:
        # distibute training
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(timeout=60, callbacks=[TrainFileLogger(enable_log=True, enable_checkpoint=False)])
        parameter = mp.train(config, mp_config)

    # --- test
    config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
    sequence.play(config, parameter)

    # --- rendering
    config.set_play_config(max_episodes=1, callbacks=[RenderingEpisode(step_stop=True)])
    sequence.play(config, parameter)


if __name__ == "__main__":

    main_use_runner(is_mp=False)
    # main_use_runner(is_mp=True)
