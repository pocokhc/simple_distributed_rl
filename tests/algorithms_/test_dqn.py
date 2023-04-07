import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    import srl.envs.ox  # noqa F401
    from srl.algorithms import dqn
    from srl.rl.models.tf.r2d3_image_block import R2D3ImageBlock
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = dqn.Config()

    def test_Pendulum(self):
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(64, 64))
        self.rl_config.enable_double_dqn = False
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 100)

    def test_Pendulum_mp(self):
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(64, 64))
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(64, 64))
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 70)

    def test_Pendulum_window(self):
        self.rl_config.window_length = 4
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(64, 64))
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 70)

    def test_OX(self):
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(128,))
        self.rl_config.epsilon = 0.5
        self.tester.verify_2player("OX", self.rl_config, 10000)

    def test_image_r2d3(self):
        env_config = srl.EnvConfig("Grid")
        self.rl_config.cnn_block = R2D3ImageBlock
        self.rl_config.hidden_block_kwargs = dict(layer_sizes=(16, 16))
        self.rl_config.change_observation_render_image = True
        self.tester.verify_1player(env_config, self.rl_config, train_count=200 * 100)


if __name__ == "__main__":
    from srl.utils import common
    common.logger_print()
    
    unittest.main(module=__name__, defaultTest="Test.test_image_r2d3", verbosity=2)
