from srl import rl
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress, RenderingEpisode

from env import my_env  # noqa F402
from env import my_env_gym  # noqa F402
from rl import my_rl_discrete_action, my_rl_table  # noqa F402


def main():

    config = sequence.Config(
        env_name="MyEnv-v0",
        # env_name="MyEnvGym-v0",
        rl_config=my_rl_table.Config(),
        # rl_config=my_rl_discrete_action.Config(),
        memory_config=rl.memory.replay_memory.Config(),
    )

    # train
    config.set_play_config(timeout=10, training=True, callbacks=[PrintProgress()])
    episode_rewards, parameter, memory = sequence.play(config)

    # test
    config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
    sequence.play(config, parameter)

    # rendering
    config.set_play_config(max_episodes=1, callbacks=[RenderingEpisode(step_stop=True)])
    sequence.play(config, parameter)


if __name__ == "__main__":
    main()
