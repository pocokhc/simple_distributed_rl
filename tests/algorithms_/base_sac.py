from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import sac

        return sac.Config()

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 25)
