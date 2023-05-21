import pytest

from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import alphazero
        from srl.rl.models.alphazero.alphazero_image_block_config import AlphaZeroImageBlockConfig
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        return alphazero.Config(
            num_simulations=100,
            sampling_steps=1,
            batch_size=32,
            discount=1.0,
            lr_schedule=[
                {"train": 0, "lr": 0.02},
                {"train": 100, "lr": 0.002},
                {"train": 1000, "lr": 0.0002},
            ],
            input_image_block=AlphaZeroImageBlockConfig(
                n_blocks=1,
                filters=32,
            ),
            value_block=MLPBlockConfig(
                layer_sizes=(32,),
            ),
            # "framework": framework, # TODO
        )

    def test_Grid(self):
        from srl.envs import grid

        config, rl_config, tester = self.create_config("Grid")
        rl_config.discount = 0.9
        rl_config.memory_warmup_size = 100
        rl_config.processors = [grid.LayerProcessor()]
        tester.train_eval(config, 1000)

    def test_StoneTaking(self):
        config, rl_config, tester = self.create_config("StoneTaking")
        config.seed = 2
        parameter, _, _ = tester.train(config, 300)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.9, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.7])

    def test_OX(self):
        config, rl_config, tester = self.create_config("OX")
        parameter, _, _ = tester.train(config, 200)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.6])

    def test_OX_mp(self):
        config, rl_config, tester = self.create_config("OX")
        config.seed = 2
        parameter, _, _ = tester.train(config, 200, is_mp=True)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def test_Othello4x4(self):
        from srl.envs import othello
        from srl.rl.models.alphazero.alphazero_image_block_config import AlphaZeroImageBlockConfig
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        config, rl_config, tester = self.create_config("Othello4x4")
        rl_config.batch_size = 32
        rl_config.memory_warmup_size = 500
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.001},
            {"train": 1000, "lr": 0.0005},
            {"train": 5000, "lr": 0.0002},
        ]
        rl_config.input_image_block = AlphaZeroImageBlockConfig(
            n_blocks=9,
            filters=32,
        )
        rl_config.value_block = MLPBlockConfig(
            layer_sizes=(16, 16),
        )
        rl_config.policy_block = MLPBlockConfig(
            layer_sizes=(32,),
        )
        rl_config.processors = [othello.LayerProcessor()]
        parameter, _, _ = tester.train(config, 20_000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.1, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.5])


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
