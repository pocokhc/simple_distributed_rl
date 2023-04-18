import pytest

from srl import runner
from srl.test import TestRL
from srl.utils import common
from srl.utils.common import is_available_gpu_tf, is_available_gpu_torch, is_package_installed

common.logger_print()

try:
    import srl.envs.ox  # noqa F401
    from srl.algorithms import dqn
    from srl.rl.models.mlp_block_config import MLPBlockConfig
except ModuleNotFoundError:
    pass


class _BaseCase:
    def return_params(self):
        raise NotImplementedError()

    def create_config(self, env):
        framework, device = self.return_params()
        rl_config = dqn.Config(
            hidden_block_config=MLPBlockConfig(layer_sizes=(64, 64)),
            enable_double_dqn=False,
            framework=framework,
        )
        config = runner.Config(env, rl_config, device=device)
        return config, rl_config

    def test_Pendulum(self):
        tester = TestRL()
        config, _ = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 100)

    def test_Pendulum_mp(self):
        tester = TestRL()
        config, _ = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        tester = TestRL()
        config, rl_config = self.create_config("Pendulum-v1")
        rl_config.enable_double_dqn = True
        tester.train(config, 200 * 70)

    def test_Pendulum_window(self):
        tester = TestRL()
        config, rl_config = self.create_config("Pendulum-v1")
        rl_config.window_length = 4
        tester.train(config, 200 * 70)

        config.model_summary()

    def test_OX(self):
        tester = TestRL()
        config, rl_config = self.create_config("OX")
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128,))
        rl_config.epsilon = 0.5
        parameter = tester.train(config, 10000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def case_image_r2d3(self):
        from srl.rl.models.r2d3_image_block_config import R2D3ImageBlockConfig

        tester = TestRL()
        config, rl_config = self.create_config("Grid")

        rl_config.image_block_config = R2D3ImageBlockConfig()
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128, 16, 16))
        rl_config.change_observation_render_image = True

        tester.train(config, 200 * 100)


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
class TestTF_CPU(_BaseCase):
    def return_params(self):
        return "tensorflow", "CPU"


@pytest.mark.skipif(not (is_package_installed("tensorflow") and is_available_gpu_tf()), reason="no module")
class TestTF_GPU(_BaseCase):
    def return_params(self):
        return "tensorflow", "GPU"

    def test_image_r2d3(self):
        self.case_image_r2d3()


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
class TestTorchCPU(_BaseCase):
    def return_params(self):
        return "torch", "CPU"


@pytest.mark.skipif(not (is_package_installed("torch") and is_available_gpu_torch()), reason="no module")
class TestTorchGPU(_BaseCase):
    def return_params(self):
        return "torch", "GPU"

    def test_image_r2d3(self):
        self.case_image_r2d3()
