import pytest

from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import agent57

        return agent57.Config(
            lstm_units=128,
            hidden_layer_sizes=(128,),
            enable_dueling_network=False,
            memory_name="ReplayMemory",
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
            # "framework": framework, # TODO
        )

    def test_Pendulum(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_mp(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 50, is_mp=True)

    def test_Pendulum_retrace(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.enable_retrace = True
        tester.train_eval(config, 200 * 50)

    def test_Pendulum_uvfa(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        tester.train_eval(config, 200 * 150)

    def test_Pendulum_memory(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.memory_name = "ProportionalMemory"
        rl_config.memory_beta_steps = 200 * 30
        tester.train_eval(config, 200 * 60)

    def test_Pendulum_dis_int(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.enable_intrinsic_reward = False
        tester.train_eval(config, 200 * 50)


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
