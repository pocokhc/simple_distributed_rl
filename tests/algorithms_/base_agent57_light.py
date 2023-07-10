from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import agent57_light
        from srl.rl.memories.config import ReplayMemoryConfig

        return agent57_light.Config(
            hidden_layer_sizes=(64, 64, 64),
            enable_dueling_network=False,
            memory=ReplayMemoryConfig(),
            target_model_update_interval=100,
            q_ext_lr=0.001,
            q_int_lr=0.001,
            actor_num=1,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
            framework=self.get_framework(),
        )

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 50)

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 100, is_mp=True)

    def test_Pendulum_uvfa(self):
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 30)
