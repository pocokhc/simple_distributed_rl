from srl.rl.memories.config import ProportionalMemoryConfig, ReplayMemoryConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import agent57

        return agent57.Config(
            lstm_units=128,
            hidden_layer_sizes=(128,),
            enable_dueling_network=False,
            memory=ReplayMemoryConfig(),
            target_model_update_interval=100,
            enable_rescale=True,
            q_ext_lr=0.001,
            q_int_lr=0.001,
            batch_size=32,
            burnin=5,
            sequence_length=10,
            enable_retrace=False,
            actor_num=8,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
        )

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 50, is_mp=True)

    def test_Pendulum_retrace(self):
        rl_config = self._create_rl_config()
        rl_config.enable_retrace = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 50)

    def test_Pendulum_uvfa(self):
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 150)

    def test_Pendulum_memory(self):
        rl_config = self._create_rl_config()
        rl_config.memory = ProportionalMemoryConfig(beta_steps=200 * 30)
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 60)

    def test_Pendulum_dis_int(self):
        rl_config = self._create_rl_config()
        rl_config.enable_intrinsic_reward = False
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 50)
