import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_agent57(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import agent57

        self.rl_config = agent57.Config()
        self.rl_config.memory_warmup_size = 100


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_agent57_light(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import agent57_light

        self.rl_config = agent57_light.Config()
        self.rl_config.memory_warmup_size = 100


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_agent57_stateful(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import agent57_stateful

        self.rl_config = agent57_stateful.Config()
        self.rl_config.memory_warmup_size = 100


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_alphazero(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import alphazero

        self.rl_config = alphazero.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_c51(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import c51

        self.rl_config = c51.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_ddpg(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import ddpg

        self.rl_config = ddpg.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_dqn(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import dqn

        self.rl_config = dqn.Config()

    def test_simple_check_atari_config(self):
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)


@unittest.skipUnless(is_package_installed("torch"), "no module")
class Test_dqn_torch(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import dqn_torch

        self.rl_config = dqn_torch.Config()


@unittest.skipUnless(
    is_package_installed("tensorflow") and is_package_installed("tensorflow_probability"), "no module"
)
class Test_dreamer(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import dreamer

        self.rl_config = dreamer.Config(
            memory_warmup_size=100,
            deter_size=10,
            stoch_size=5,
            num_units=10,
            cnn_depth=4,
            batch_size=4,
            batch_length=5,
        )
        self.rl_config.change_observation_render_image = True


class Test_dynaq(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import dynaq

        self.rl_config = dynaq.Config()


class Test_mcts(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import mcts

        self.rl_config = mcts.Config()
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
            train_kwargs=dict(max_steps=10),
        )


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_muzero(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import muzero

        self.rl_config = muzero.Config()
        self.rl_config.set_parameter(
            dict(
                batch_size=4,
                memory_warmup_size=20,
            )
        )
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
        )


@unittest.skipUnless(
    is_package_installed("tensorflow") and is_package_installed("tensorflow_probability"), "no module"
)
class Test_planet(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import planet

        self.rl_config = planet.Config()
        self.rl_config.set_parameter(
            dict(
                batch_size=4,
                batch_length=5,
                memory_warmup_size=100,
                num_generation=2,
                num_simulations=2,
                print_ga_debug=False,
            )
        )
        self.rl_config.change_observation_render_image = True


class Test_ql_agent57(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import ql_agent57

        self.rl_config = ql_agent57.Config()


class Test_ql(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import ql

        self.rl_config = ql.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_r2d2(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import r2d2

        self.rl_config = r2d2.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_rainbow(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import rainbow

        self.rl_config = rainbow.Config()

    @unittest.skipUnless(is_package_installed("tensorflow_addons"), "no module")
    def test_simple_check_atari_config(self):
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_sac(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import sac

        self.rl_config = sac.Config()


class Test_search_dynaq(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import search_dynaq

        self.rl_config = search_dynaq.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_stochastic_muzero(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import stochastic_muzero

        self.rl_config = stochastic_muzero.Config()
        self.rl_config.set_parameter(
            dict(
                batch_size=4,
                memory_warmup_size=20,
                dynamics_blocks=1,
                input_image_block_kwargs=dict(
                    n_blocks=1,
                    filters=64,
                ),
            )
        )
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
        )


class Test_vanilla_policy(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import vanilla_policy

        self.rl_config = vanilla_policy.Config()


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test_world_models(TestRL):
    def setUp(self) -> None:
        from srl.algorithms import world_models

        self.rl_config = world_models.Config()


if __name__ == "__main__":
    from srl.utils import common

    common.logger_print()

    # unittest.main(module=__name__, defaultTest="Test_planet.test_simple_check", verbosity=2)
    # unittest.main(module=__name__, defaultTest="Test_dreamer.test_simple_check_mp", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test_muzero.test_summary", verbosity=2)
