from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import sac

        rl_config = sac.Config(
            lr_policy=0.003,
            lr_q=0.003,
            lr_alpha=0.001,
        )
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 25)
        tester.eval(runner)
