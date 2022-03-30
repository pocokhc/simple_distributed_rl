import unittest

from srl import rl
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, Rendering
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


class Test(unittest.TestCase):
    def test_run(self):
        main_use_runner(False)
        main_use_runner(True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_run", verbosity=2)
