import pytest

from srl.base.define import RLTypes

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import ppo

        rl_config = ppo.Config(
            batch_size=128,
            memory_warmup_size=1000,
            discount=0.9,
            optimizer_initial_lr=0.01,
            optimizer_final_lr=0.01,
            surrogate_type="clip",
            baseline_type="normal",
            experience_collection_method="MC",
            enable_value_clip=False,
            enable_state_normalized=False,
        )
        rl_config.hidden_block.set_mlp((32, 32))
        rl_config.memory.capacity = 1000
        return rl_config

    def test_Grid(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    @pytest.mark.parametrize("baseline_type", ["", "ave", "std", "advantage"])
    def test_Grid_baseline(self, baseline_type):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.baseline_type = baseline_type
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    def test_Grid_GAE(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    def test_Grid_KL(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.surrogate_type = "KL"
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    def test_EasyGrid_Continuous(self):
        self.check_skip()
        # 学習できない… TODO
        rl_config = self._create_rl_config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        rl_config.hidden_block.set_mlp((64, 64, 64))
        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=100_000)
        tester.eval(runner)
