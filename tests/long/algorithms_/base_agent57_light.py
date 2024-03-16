from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config(
            target_model_update_interval=100,
            lr_ext=0.001,
            lr_int=0.001,
            actor_num=2,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
        )
        rl_config.dueling_network.set((64, 64), True)
        rl_config.memory.set_replay_memory()

        if self.get_framework() == "tensorflow":
            rl_config.framework.set_tensorflow()
        elif self.get_framework() == "torch":
            rl_config.framework.set_torch()
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        tester.eval(runner)

    def test_Pendulum_uvfa(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 50)
        tester.eval(runner)

    def test_OX(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((32, 32, 16), False)

        runner, tester = self.create_runner("OX", rl_config)
        runner.train(max_train_count=10_000)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.4, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.4])
