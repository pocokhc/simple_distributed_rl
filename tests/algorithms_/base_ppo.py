import pytest

from srl.base.define import RLTypes

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import ppo

        rl_config = ppo.Config(
            batch_size=128,
            discount=0.9,
            surrogate_type="clip",
            baseline_type="normal",
            experience_collection_method="MC",
            enable_value_clip=False,
            enable_state_normalized=False,
        )
        rl_config.lr.set_linear(1000, 0.01, 0.001)
        rl_config.hidden_block.set_mlp((32, 32))
        rl_config.memory.capacity = 1000
        rl_config.memory.warmup_size = 1000
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

    def test_Pendulum(self):
        self.check_skip()
        # うまく学習できない... TODO
        rl_config = self._create_rl_config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        rl_config.lr.set_constant(0.0001)
        rl_config.value_loss_weight = 0.5
        rl_config.entropy_weight = 0.0001
        rl_config.hidden_block.set_mlp((128,))
        rl_config.policy_block.set_mlp((128, 128))
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 500)
        tester.eval(runner)
