from . import (
    base_agent57,
    base_agent57_light,
    base_alphazero,
    base_c51,
    base_ddpg,
    base_dqn,
    base_dreamer_v3,
    base_muzero,
    base_planet,
    base_ppo,
    base_r2d2,
    base_rainbow,
    base_sac,
    base_stochastic_muzero,
    base_world_models,
)


class Test_agent57_light_CPU(base_agent57_light.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_agent57_light_GPU(base_agent57_light.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_agent57_CPU(base_agent57.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_agent57_GPU(base_agent57.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_alphazero_CPU(base_alphazero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_alphazero_GPU(base_alphazero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_c51_CPU(base_c51.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_c51_GPU(base_c51.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_ddpg_CPU(base_ddpg.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_ddpg_GPU(base_ddpg.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_dqn_CPU(base_dqn.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_dqn_GPU(base_dqn.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_dreamer_v3_CPU(base_dreamer_v3.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_dreamer_v3_GPU(base_dreamer_v3.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_muzero_CPU(base_muzero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_muzero_GPU(base_muzero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_planet_CPU(base_planet.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_planet_GPU(base_planet.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_ppo_CPU(base_ppo.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_ppo_GPU(base_ppo.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_r2d2_CPU(base_r2d2.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_r2d2_GPU(base_r2d2.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_rainbow_CPU(base_rainbow.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_rainbow_GPU(base_rainbow.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_sac_CPU(base_sac.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_sac_GPU(base_sac.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_stochastic_muzero_CPU(base_stochastic_muzero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_stochastic_muzero_GPU(base_stochastic_muzero.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_world_models_CPU(base_world_models.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class Test_world_models_GPU(base_world_models.BaseCase):
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"
