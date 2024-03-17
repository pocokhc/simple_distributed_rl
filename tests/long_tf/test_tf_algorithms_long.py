from tests.algorithms_ import (
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


class _CommonOption_CPU:
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "CPU"


class _CommonOption_GPU:
    def get_framework(self) -> str:
        return "tensorflow"

    def get_device(self) -> str:
        return "GPU"


class Test_agent57_light_CPU(_CommonOption_CPU, base_agent57_light.BaseCase):
    pass


class Test_agent57_light_GPU(_CommonOption_GPU, base_agent57_light.BaseCase):
    pass


class Test_agent57_CPU(_CommonOption_CPU, base_agent57.BaseCase):
    pass


class Test_agent57_GPU(_CommonOption_GPU, base_agent57.BaseCase):
    pass


class Test_alphazero_CPU(_CommonOption_CPU, base_alphazero.BaseCase):
    pass


class Test_alphazero_GPU(_CommonOption_GPU, base_alphazero.BaseCase):
    pass


class Test_c51_CPU(_CommonOption_CPU, base_c51.BaseCase):
    pass


class Test_c51_GPU(_CommonOption_GPU, base_c51.BaseCase):
    pass


class Test_ddpg_CPU(_CommonOption_CPU, base_ddpg.BaseCase):
    pass


class Test_ddpg_GPU(_CommonOption_GPU, base_ddpg.BaseCase):
    pass


class Test_dqn_CPU(_CommonOption_CPU, base_dqn.BaseCase):
    pass


class Test_dqn_GPU(_CommonOption_GPU, base_dqn.BaseCase):
    pass


class Test_dreamer_v3_CPU(_CommonOption_CPU, base_dreamer_v3.BaseCase):
    pass


class Test_dreamer_v3_GPU(_CommonOption_GPU, base_dreamer_v3.BaseCase):
    pass


class Test_muzero_CPU(_CommonOption_CPU, base_muzero.BaseCase):
    pass


class Test_muzero_GPU(_CommonOption_GPU, base_muzero.BaseCase):
    pass


class Test_planet_CPU(_CommonOption_CPU, base_planet.BaseCase):
    pass


class Test_planet_GPU(_CommonOption_GPU, base_planet.BaseCase):
    pass


class Test_ppo_CPU(_CommonOption_CPU, base_ppo.BaseCase):
    pass


class Test_ppo_GPU(_CommonOption_GPU, base_ppo.BaseCase):
    pass


class Test_r2d2_CPU(_CommonOption_CPU, base_r2d2.BaseCase):
    pass


class Test_r2d2_GPU(_CommonOption_GPU, base_r2d2.BaseCase):
    pass


class Test_rainbow_CPU(_CommonOption_CPU, base_rainbow.BaseCase):
    pass


class Test_rainbow_GPU(_CommonOption_GPU, base_rainbow.BaseCase):
    pass


class Test_sac_CPU(_CommonOption_CPU, base_sac.BaseCase):
    pass


class Test_sac_GPU(_CommonOption_GPU, base_sac.BaseCase):
    pass


class Test_stochastic_muzero_CPU(_CommonOption_CPU, base_stochastic_muzero.BaseCase):
    pass


class Test_stochastic_muzero_GPU(_CommonOption_GPU, base_stochastic_muzero.BaseCase):
    pass


class Test_world_models_CPU(_CommonOption_CPU, base_world_models.BaseCase):
    pass


class Test_world_models_GPU(_CommonOption_GPU, base_world_models.BaseCase):
    pass
