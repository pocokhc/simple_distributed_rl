from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import sac

        rl_config = sac.Config()
        return rl_config

    def test_EasyGrid(self):
        self.check_skip()
        rl_config = self._create_rl_config()

        rl_config.batch_size = 32
        rl_config.lr_policy.set_constant(0.0002)
        rl_config.lr_q.set_constant(0.001)
        rl_config.memory.capacity = 10000
        rl_config.memory.warmup_size = 1000
        rl_config.policy_hidden_block.set_mlp((32, 32, 32))
        rl_config.q_hidden_block.set_mlp((32, 32, 32))
        rl_config.entropy_bonus_exclude_q = True
        rl_config.entropy_alpha = 0.1
        rl_config.entropy_alpha_auto_scale = False

        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=4000)
        tester.eval(runner)

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)

        rl_config.batch_size = 32
        rl_config.lr_policy.set_constant(0.003)
        rl_config.lr_q.set_constant(0.003)
        rl_config.memory.capacity = 10000
        rl_config.memory.warmup_size = 1000
        rl_config.policy_hidden_block.set_mlp((64, 64, 64))
        rl_config.q_hidden_block.set_mlp((128, 128, 128))

        runner.train(max_train_count=200 * 30)
        tester.eval(runner)
