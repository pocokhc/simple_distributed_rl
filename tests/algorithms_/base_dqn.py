from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import dqn
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        return dqn.Config(
            hidden_block_config=MLPBlockConfig(layer_sizes=(64, 64)),
            enable_double_dqn=False,
            framework=self.get_framework(),
        )

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 100)

    def test_Pendulum_mp(self):
        from srl.rl import memories

        rl_config = self._create_rl_config()
        rl_config.memory = memories.ProportionalMemoryConfig()

        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        rl_config = self._create_rl_config()
        rl_config.enable_double_dqn = True

        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train(config, 200 * 70)

    def test_Pendulum_window(self):
        rl_config = self._create_rl_config()
        rl_config.window_length = 4

        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train(config, 200 * 70)

        config.model_summary()

    def test_OX(self):
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        rl_config = self._create_rl_config()
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128,))
        rl_config.epsilon = 0.5

        config, tester = self.create_config("OX", rl_config)
        config.seed = 2
        parameter, _, _ = tester.train(config, 10000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def case_image_r2d3(self):
        from srl.rl.models.dqn.r2d3_image_block_config import R2D3ImageBlockConfig
        from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig

        rl_config = self._create_rl_config()
        rl_config.image_block_config = R2D3ImageBlockConfig()
        rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(128, 16, 16))
        rl_config.use_render_image_for_observation = True

        config, tester = self.create_config("Grid", rl_config)
        tester.train(config, 200 * 100)
