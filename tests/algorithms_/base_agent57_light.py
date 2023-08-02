from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config(
            hidden_layer_sizes=(64, 64, 64),
            enable_dueling_network=False,
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
        rl_config.memory.set_replay_memory()
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 50)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_steps=200 * 100)
        tester.eval(runner)

    def test_Pendulum_uvfa(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 30)
        tester.eval(runner)
