from srl.base.define import RLTypes

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import ppo

        rl_config = ppo.Config(
            batch_size=32,
            discount=0.9,
            gae_discount=0.9,
            surrogate_type="clip",
            baseline_type="normal",
            experience_collection_method="MC",
            enable_value_clip=False,
            enable_state_normalized=False,
        )
        rl_config.lr.set_constant(0.005)
        rl_config.hidden_block.set_mlp((64, 64))
        rl_config.value_block.set_mlp(())
        rl_config.policy_block.set_mlp(())
        rl_config.memory.capacity = 1000
        rl_config.memory.warmup_size = 1000
        return rl_config

    def test_EasyGrid1(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        rl_config.baseline_type = ""
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = True
        rl_config.enable_state_normalized = False
        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    def test_Grid2(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        rl_config.baseline_type = "v"
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = False
        rl_config.enable_state_normalized = False
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)

    def test_Grid3(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "MC"
        rl_config.baseline_type = "normal"
        rl_config.surrogate_type = "kl"
        rl_config.enable_value_clip = False
        rl_config.enable_state_normalized = False
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=30000)
        tester.eval(runner)

    def test_Grid4(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "MC"
        rl_config.baseline_type = "normal"
        rl_config.surrogate_type = ""  # ""は学習がそもそも難しい
        rl_config.enable_value_clip = True
        rl_config.enable_state_normalized = True
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=20000)
        tester.eval(runner, baseline=-1)

    def test_EasyGrid_continue(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.lr.set_constant(0.001)
        rl_config.experience_collection_method = "GAE"
        rl_config.baseline_type = ""
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = False
        rl_config.enable_state_normalized = False
        rl_config.entropy_weight = 1.0
        rl_config.override_action_type = RLTypes.CONTINUOUS
        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=40000)
        tester.eval(runner)
