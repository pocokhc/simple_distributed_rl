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
        return "torch"

    def get_device(self) -> str:
        return "CPU"


class _CommonOption_GPU:
    def get_framework(self) -> str:
        return "torch"

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


class Test_dqn_CPU(_CommonOption_CPU, base_dqn.BaseCase):
    pass


class Test_dqn_GPU(_CommonOption_GPU, base_dqn.BaseCase):
    pass


class Test_planet_CPU(_CommonOption_CPU, base_planet.BaseCase):
    pass


class Test_planet_GPU(_CommonOption_GPU, base_planet.BaseCase):
    pass


class Test_rainbow_CPU(_CommonOption_CPU, base_rainbow.BaseCase):
    pass


class Test_rainbow_GPU(_CommonOption_GPU, base_rainbow.BaseCase):
    pass
