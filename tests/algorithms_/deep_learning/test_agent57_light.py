import pytest

from srl.rl.memories.config import ReplayMemoryConfig
from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import agent57_light

        return agent57_light.Config(
            hidden_layer_sizes=(64, 64, 64),
            enable_dueling_network=False,
            memory=ReplayMemoryConfig(),
            target_model_update_interval=100,
            q_ext_lr=0.001,
            q_int_lr=0.001,
            actor_num=1,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
            # framework = framework, # TODO
        )

    def test_Pendulum(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 50)

    def test_Pendulum_mp(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 100, is_mp=True)

    def test_Pendulum_uvfa(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        tester.train_eval(config, 200 * 30)

    def test_Pendulum_memory(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 40)


class TestTF_CPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")

        return "tensorflow", "CPU"


class TestTF_GPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")
        if not common.is_available_gpu_tf():
            pytest.skip()

        return "tensorflow", "GPU"
