import pytest

from srl.rl.memories.config import ReplayMemoryConfig
from srl.utils import common

from .common_base_class import CommonBaseClass

common.logger_print()


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import muzero

        return muzero.Config(
            batch_size=16,
            memory_warmup_size=50,
        )

    def test_EasyGrid(self):
        from srl.envs import grid
        from srl.rl.models.alphazero import AlphaZeroImageBlockConfig

        config, rl_config, tester = self.create_config("EasyGrid")
        rl_config.set_parameter(
            dict(
                num_simulations=20,
                discount=0.9,
                batch_size=16,
                memory_warmup_size=200,
                memory=ReplayMemoryConfig(),
                lr_init=0.002,
                lr_decay_steps=10_000,
                v_min=-2,
                v_max=2,
                unroll_steps=1,
                input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
                dynamics_blocks=1,
                enable_rescale=False,
                weight_decay=0,
            )
        )
        rl_config.processors = [grid.LayerProcessor()]
        tester.train_eval(config, 2000)

    def test_EasyGrid_PER(self):
        from srl.envs import grid
        from srl.rl.models.alphazero import AlphaZeroImageBlockConfig

        config, rl_config, tester = self.create_config("EasyGrid")
        rl_config.set_parameter(
            dict(
                num_simulations=20,
                discount=0.9,
                batch_size=16,
                memory_warmup_size=200,
                lr_init=0.002,
                lr_decay_steps=10_000,
                v_min=-2,
                v_max=2,
                unroll_steps=1,
                input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
                dynamics_blocks=1,
                enable_rescale=False,
                weight_decay=0,
            )
        )
        rl_config.processors = [grid.LayerProcessor()]
        tester.train_eval(config, 3000)


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
