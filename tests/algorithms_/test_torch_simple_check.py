from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_agent57(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import agent57

        rl_config = agent57.Config()
        rl_config.framework.set_torch()

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
        pytest.importorskip("torch")

        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config()
        rl_config.framework.set_torch()

        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.image_block.set_dqn_image(2)
        rl_config.dueling_network.set((2, 2), enable=True)
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True

        return rl_config, {}


class Test_dqn(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_torch()

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


class Test_rainbow(CommonBaseSimpleTest):
    @pytest.fixture(params=[1, 2])
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_torch()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.multisteps = rl_param
        rl_config.target_model_update_interval = 1
        rl_config.enable_rescale = True

        return rl_config, {}
