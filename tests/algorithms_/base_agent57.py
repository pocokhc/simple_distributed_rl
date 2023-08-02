from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import agent57

        rl_config = agent57.Config(
            lstm_units=128,
            hidden_layer_sizes=(128,),
            enable_dueling_network=False,
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
        rl_config.memory.set_replay_memory()
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70, enable_eval=False)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 50, enable_eval=False)
        tester.eval(runner)

    def test_Pendulum_retrace(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.enable_retrace = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50, enable_eval=False)
        tester.eval(runner)

    def test_Pendulum_uvfa(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 150, enable_eval=False)
        tester.eval(runner)

    def test_Pendulum_memory(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.memory.set_proportional_memory(beta_steps=200 * 30)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 60, enable_eval=False)
        tester.eval(runner)

    def test_Pendulum_dis_int(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.enable_intrinsic_reward = False
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50, enable_eval=False)
        tester.eval(runner)
