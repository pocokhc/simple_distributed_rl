import pytest

from srl.base.define import RLTypes
from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import ppo
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        return ppo.Config(
            batch_size=128,
            hidden_block=MLPBlockConfig((32, 32)),
            memory_warmup_size=1000,
            capacity=1000,
            discount=0.9,
            optimizer_initial_lr=0.01,
            optimizer_final_lr=0.01,
            surrogate_type="clip",
            baseline_type="normal",
            experience_collection_method="MC",
            enable_value_clip=False,
            enable_state_normalized=False,
        )

    def test_Grid(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 10000)

    @pytest.mark.parametrize("baseline_type", ["", "ave", "std", "advantage"])
    def test_Grid_baseline(self, baseline_type):
        rl_config = self._create_rl_config()
        rl_config.baseline_type = baseline_type
        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 10000)

    def test_Grid_GAE(self):
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 10000)

    def test_Grid_KL(self):
        rl_config = self._create_rl_config()
        rl_config.surrogate_type = "KL"
        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 10000)

    def test_EasyGrid_Continuous(self):
        rl_config = self._create_rl_config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(64, 64, 64))
        config, tester = self.create_config("EasyGrid", rl_config)
        tester.train_eval(config, 100_000)
