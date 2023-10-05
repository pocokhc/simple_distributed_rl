from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_agent57(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57

        rl_config = agent57.Config()
        rl_config.framework.set_tensorflow()
        rl_config.memory_warmup_size = 100
        return rl_config, {}


class Test_agent57_light(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config()
        rl_config.framework.set_tensorflow()
        rl_config.memory_warmup_size = 100
        return rl_config, {}


class Test_agent57_stateful(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57_stateful

        rl_config = agent57_stateful.Config()
        rl_config.memory_warmup_size = 100
        return rl_config, {}


class Test_alphazero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import alphazero

        rl_config = alphazero.Config()
        return rl_config, {}

    def test_simple_check_go_config(self):
        pytest.importorskip("tensorflow")
        from srl.algorithms import alphazero

        rl_config = alphazero.Config()
        rl_config.set_go_config()
        rl_config.memory_warmup_size = 100
        rl_config.batch_size = 8
        rl_config.num_simulations = 5

        tester = TestRL()
        tester.simple_check(rl_config)


class Test_c51(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import c51

        rl_config = c51.Config()
        return rl_config, {}


class Test_ddpg(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import ddpg

        rl_config = ddpg.Config()
        return rl_config, {}


class Test_dqn(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_tensorflow()
        return rl_config, {}

    def test_simple_check_atari_config(self):
        pytest.importorskip("tensorflow")
        pytest.importorskip("pygame")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_tensorflow()
        rl_config.use_render_image_for_observation = True
        rl_config.set_atari_config()
        rl_config.memory_warmup_size = 1000

        tester = TestRL()
        tester.simple_check(rl_config)


class Test_dreamer(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import dreamer

        rl_config = dreamer.Config(
            memory_warmup_size=10,
            deter_size=10,
            stoch_size=5,
            reward_layer_sizes=(10, 10),
            critic_layer_sizes=(10, 10),
            actor_layer_sizes=(10, 10),
            cnn_depth=2,
            batch_size=2,
            batch_length=5,
        )
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_dreamer_v2(CommonBaseSimpleTest):
    @pytest.fixture(params=["simple", "dreamer", "dreamer_v2"])
    def critic_estimation_method(self, request):
        return request.param

    @pytest.fixture(params=["episode", "loop", "episode_steps"])
    def experience_acquisition_method(self, request):
        return request.param

    @pytest.fixture()
    def rl_param(self, critic_estimation_method, experience_acquisition_method):
        return [critic_estimation_method, experience_acquisition_method]

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import dreamer_v2

        rl_config = dreamer_v2.Config(
            memory_warmup_size=10,
            deter_size=10,
            stoch_size=5,
            reward_layer_sizes=(5, 5),
            discount_layer_sizes=(5, 5),
            critic_layer_sizes=(5, 5),
            actor_layer_sizes=(5, 5),
            cnn_depth=2,
            batch_size=2,
            batch_length=5,
            critic_estimation_method=rl_param[0],
            experience_acquisition_method=rl_param[1],
        )
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_muzero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import muzero

        rl_config = muzero.Config(
            batch_size=4,
            num_simulations=3,
            memory_warmup_size=10,
            dynamics_blocks=1,
        )
        rl_config.input_image_block.set_alphazero_block(2, 4)
        return rl_config, dict(use_layer_processor=True)

    def test_simple_check_atari_config(self):
        pytest.importorskip("tensorflow")
        pytest.importorskip("ale_py")

        from srl.algorithms import muzero

        rl_config = muzero.Config(
            batch_size=4,
            num_simulations=3,
            memory_warmup_size=10,
            dynamics_blocks=1,
        )
        rl_config.input_image_block.set_alphazero_block(2, 4)
        rl_config.set_atari_config()
        rl_config.memory_warmup_size = 5
        rl_config.batch_size = 2
        rl_config.num_simulations = 2

        tester = TestRL()
        tester.simple_check(
            rl_config,
            env_list=["ALE/Pong-v5"],
            train_kwargs={"max_steps": 10},
            use_layer_processor=True,
        )


class Test_planet(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import planet

        rl_config = planet.Config(
            batch_size=4,
            batch_length=5,
            memory_warmup_size=100,
            num_generation=2,
            num_simulations=2,
            print_ga_debug=False,
        )
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_ppo(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import ppo

        rl_config = ppo.Config()
        return rl_config, {}


class Test_r2d2(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import r2d2

        rl_config = r2d2.Config()
        return rl_config, {}


class Test_rainbow(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_tensorflow()
        return rl_config, {}

    def test_simple_check_atari_config(self):
        pytest.importorskip("tensorflow")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_tensorflow()
        rl_config.set_atari_config()
        rl_config.memory_warmup_size = 1000

        tester = TestRL()
        tester.simple_check(rl_config)


class Test_sac(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import sac

        rl_config = sac.Config()
        return rl_config, {}


class Test_stochastic_muzero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import stochastic_muzero

        rl_config = stochastic_muzero.Config(
            num_simulations=3,
            batch_size=2,
            memory_warmup_size=5,
            dynamics_blocks=1,
        )
        rl_config.input_image_block.set_alphazero_block(1, 8)
        return rl_config, dict(use_layer_processor=True)


class Test_world_models(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import world_models

        rl_config = world_models.Config()
        return rl_config, {}
