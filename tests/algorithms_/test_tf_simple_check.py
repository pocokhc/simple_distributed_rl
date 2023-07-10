import unittest

import pytest

from srl.test import TestRL
from srl.utils import common

common.logger_print()


class Test_agent57(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57

        self.rl_config = agent57.Config()
        self.rl_config.memory_warmup_size = 100


class Test_agent57_light_tf(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57_light

        self.rl_config = agent57_light.Config(framework="tensorflow")
        self.rl_config.memory_warmup_size = 100


class Test_agent57_stateful(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57_stateful

        self.rl_config = agent57_stateful.Config()
        self.rl_config.memory_warmup_size = 100


class Test_alphazero(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import alphazero

        self.rl_config = alphazero.Config()

    def test_simple_check_go_config(self):
        pytest.importorskip("tensorflow")

        self.init_simple_check()
        self.rl_config.set_go_config()
        self.rl_config.memory_warmup_size = 100
        self.rl_config.batch_size = 32
        self.rl_config.num_simulations = 50
        self.simple_check(self.rl_config)


class Test_c51(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import c51

        self.rl_config = c51.Config()


class Test_ddpg(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import ddpg

        self.rl_config = ddpg.Config()


class Test_dqn(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import dqn

        self.rl_config = dqn.Config(framework="tensorflow")

    def test_simple_check_atari_config(self):
        self.init_simple_check()
        self.rl_config.use_render_image_for_observation = True
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)


class Test_dreamer(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")

        from srl.algorithms import dreamer

        self.rl_config = dreamer.Config(
            memory_warmup_size=100,
            deter_size=10,
            stoch_size=5,
            reward_num_units=10,
            value_num_units=10,
            action_num_units=10,
            cnn_depth=4,
            batch_size=4,
            batch_length=5,
        )
        self.rl_config.use_render_image_for_observation = True


class Test_muzero(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import muzero
        from srl.rl.models.alphazero.alphazero_image_block_config import AlphaZeroImageBlockConfig

        self.rl_config = muzero.Config(
            batch_size=4,
            num_simulations=3,
            memory_warmup_size=10,
            dynamics_blocks=1,
            input_image_block=AlphaZeroImageBlockConfig(n_blocks=2, filters=8),
        )
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
        )

    def test_simple_check_atari_config(self):
        pytest.importorskip("tensorflow")
        pytest.importorskip("ale_py")

        self.init_simple_check()
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 5
        self.rl_config.batch_size = 2
        self.rl_config.num_simulations = 2
        self.simple_check(self.rl_config, env_list=["ALE/Pong-v5"], train_kwargs={"max_steps": 10})


class Test_planet(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")

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
        self.rl_config.use_render_image_for_observation = True


class Test_r2d2(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import r2d2

        self.rl_config = r2d2.Config()


class Test_rainbow_tensorflow(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import rainbow

        self.rl_config = rainbow.Config(framework="tensorflow")

    def test_simple_check_atari_config(self):
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_addons")

        self.init_simple_check()
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)


class Test_sac(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import sac

        self.rl_config = sac.Config()


class Test_stochastic_muzero(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import stochastic_muzero
        from srl.rl.models.alphazero.alphazero_image_block_config import AlphaZeroImageBlockConfig

        self.rl_config = stochastic_muzero.Config(
            num_simulations=3,
            batch_size=2,
            memory_warmup_size=5,
            dynamics_blocks=1,
            input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=8),
        )
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
        )


class Test_world_models(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("tensorflow")

        from srl.algorithms import world_models

        self.rl_config = world_models.Config()
