from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import c51

        return c51.Config()

    def test_Grid(self):
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.002
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(16, 16))
        rl_config.categorical_num_atoms = 11
        rl_config.categorical_v_min = -2
        rl_config.categorical_v_max = 2
        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 6000)

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(32, 32, 32))
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 600)

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(32, 32, 32))
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 600, is_mp=True)
