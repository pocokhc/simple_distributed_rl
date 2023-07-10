from srl.rl.memories.config import ReplayMemoryConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import r2d2

        return r2d2.Config(
            lstm_units=32,
            hidden_layer_sizes=(16, 16),
            enable_dueling_network=False,
            memory=ReplayMemoryConfig(),
            target_model_update_interval=100,
            enable_rescale=True,
            burnin=5,
            sequence_length=5,
            enable_retrace=False,
        )

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 35)

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 20, is_mp=True)

    def test_Pendulum_retrace(self):
        rl_config = self._create_rl_config()
        rl_config.enable_retrace = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 35)
