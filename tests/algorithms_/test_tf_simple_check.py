from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_agent57(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57

        rl_config = agent57.Config()
        rl_config.framework.set_tensorflow()

        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.lstm_units = 8
        rl_config.image_block.set_dqn_image(2)
        rl_config.dueling_network.set((2, 2), enable=True)
        rl_config.target_model_update_interval = 1
        rl_config.burnin = 1
        rl_config.sequence_length = 2
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True

        return rl_config, {}


class Test_agent57_light(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config()
        rl_config.framework.set_tensorflow()

        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.image_block.set_dqn_image(2)
        rl_config.dueling_network.set((2, 2), enable=True)
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True

        return rl_config, {}


class Test_alphazero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import alphazero

        rl_config = alphazero.Config()
        rl_config.set_go_config()

        rl_config.num_simulations = 2
        rl_config.sampling_steps = 2
        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.input_image_block.set_alphazero_block(1, 2)
        rl_config.value_block.set_mlp((2, 2))

        return rl_config, {}


class Test_c51(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import c51

        rl_config = c51.Config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2

        return rl_config, {}


class Test_ddpg(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import ddpg

        rl_config = ddpg.Config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2

        return rl_config, {}


class Test_dqn(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_tensorflow()

        rl_config.set_atari_config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.enable_double_dqn = True
        rl_config.enable_rescale = True
        rl_config.use_render_image_for_observation = True
        rl_config.image_block.set_dqn_image(2)
        rl_config.hidden_block.set_mlp((2,))

        return rl_config, {}


class Test_dreamer(CommonBaseSimpleTest):
    # 総当たりは時間がかかるので、どれか１つ実行に変更
    @pytest.fixture(
        params=[
            ["simple", "episode", "none"],
            ["simple", "loop", "none"],
            ["dreamer", "episode_steps", "tanh"],
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import dreamer

        rl_config = dreamer.Config(
            deter_size=2,
            stoch_size=2,
            reward_layer_sizes=(2, 2),
            critic_layer_sizes=(2, 2),
            actor_layer_sizes=(2, 2),
            cnn_depth=2,
            batch_size=2,
            batch_length=2,
            critic_estimation_method=rl_param[0],
            experience_acquisition_method=rl_param[1],
            clip_rewards=rl_param[2],
        )
        rl_config.memory.warmup_size = 2
        rl_config.horizon = 2
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_dreamer_v2(CommonBaseSimpleTest):
    # 総当たりは時間がかかるので、どれか１つ実行に変更
    @pytest.fixture(
        params=[
            ["simple", "episode", "none"],
            ["dreamer", "loop", "none"],
            ["dreamer_v2", "episode_steps", "tanh"],
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import dreamer_v2

        rl_config = dreamer_v2.Config(
            deter_size=2,
            stoch_size=2,
            reward_layer_sizes=(5, 5),
            discount_layer_sizes=(5, 5),
            critic_layer_sizes=(5, 5),
            actor_layer_sizes=(5, 5),
            cnn_depth=2,
            batch_size=2,
            batch_length=2,
            critic_estimation_method=rl_param[0],
            experience_acquisition_method=rl_param[1],
            clip_rewards=rl_param[2],
        )
        rl_config.memory.warmup_size = 2
        rl_config.horizon = 2
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_muzero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import muzero

        rl_config = muzero.Config()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.num_simulations = 2
        rl_config.unroll_steps = 2
        rl_config.input_image_block.set_alphazero_block(1, 2)
        rl_config.dynamics_blocks = 1

        return rl_config, dict(use_layer_processor=True)


class Test_planet(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import planet

        rl_config = planet.Config(
            batch_size=2,
            batch_length=2,
            pred_action_length=2,
            num_generation=2,
            num_individual=2,
            num_simulations=2,
            print_ga_debug=False,
        )
        rl_config.memory.warmup_size = 2
        rl_config.use_render_image_for_observation = True
        return rl_config, {}


class Test_ppo(CommonBaseSimpleTest):
    @pytest.fixture(
        params=[
            ["MC", "", ""],
            ["MC", "ave", "clip"],
            ["GAE", "std", "kl"],
            ["GAE", "normal", "kl"],
            ["GAE", "advantage", "kl"],
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import ppo

        rl_config = ppo.Config(
            experience_collection_method=rl_param[0],
            baseline_type=rl_param[1],
            surrogate_type=rl_param[2],
        )
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2

        return rl_config, {}


class Test_r2d2(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import r2d2

        rl_config = r2d2.Config()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.lstm_units = 2
        rl_config.burnin = 2
        rl_config.sequence_length = 2
        rl_config.target_model_update_interval = 1
        rl_config.enable_retrace = True

        return rl_config, {}


class Test_rainbow(CommonBaseSimpleTest):
    @pytest.fixture(params=[1, 2])
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_tensorflow()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.multisteps = rl_param
        rl_config.target_model_update_interval = 1
        rl_config.enable_rescale = True

        return rl_config, {}


class Test_sac(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import sac

        rl_config = sac.Config()
        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2

        return rl_config, {}


class Test_stochastic_muzero(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import stochastic_muzero

        rl_config = stochastic_muzero.Config(
            num_simulations=1,
            unroll_steps=1,
            batch_size=2,
            dynamics_blocks=1,
        )
        rl_config.memory.warmup_size = 2
        rl_config.input_image_block.set_alphazero_block(1, 4)
        return rl_config, dict(use_layer_processor=True)


class Test_world_models(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import world_models

        rl_config = world_models.Config()
        return rl_config, {}
