import pytest

from srl.utils import common

from .common_base_class import CommonBaseClass

common.logger_print()


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import dqn
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        return dqn.Config(
            hidden_block_config=MLPBlockConfig(layer_sizes=(64, 64)),
            enable_double_dqn=False,
            framework=framework,
        )

    def test_Pendulum(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 100)

    def test_Pendulum_mp(self):
        from srl.rl import memories

        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.memory = memories.ProportionalMemoryConfig()
        tester.train_eval(config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.enable_double_dqn = True
        tester.train(config, 200 * 70)

    def test_Pendulum_window(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.window_length = 4
        tester.train(config, 200 * 70)

        config.model_summary()

    def test_OX(self):
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        config, rl_config, tester = self.create_config("OX")
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128,))
        rl_config.epsilon = 0.5
        config.seed = 2
        parameter, _, _ = tester.train(config, 10000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def case_image_r2d3(self):
        from srl.rl.models.dqn.r2d3_image_block_config import R2D3ImageBlockConfig
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        config, rl_config, tester = self.create_config("Grid")

        rl_config.image_block_config = R2D3ImageBlockConfig()
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128, 16, 16))
        rl_config.change_observation_render_image = True

        tester.train(config, 200 * 100)


class TestTF_CPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")

        return "tensorflow", "CPU"


class TestTF_GPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")
        if not common.is_available_gpu_tf():
            pytest.skip()

        return "tensorflow", "GPU"

    def test_image_r2d3(self):
        self.case_image_r2d3()


class TestTorchCPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("torch")

        return "torch", "CPU"


class TestTorchGPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("torch")
        if not common.is_available_gpu_torch():
            pytest.skip()

        return "torch", "GPU"

    def test_image_r2d3(self):
        self.case_image_r2d3()
