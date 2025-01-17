from tests.algorithms_ import (
    base_agent57,
    base_agent57_light,
    base_alphazero,
    base_c51,
    base_ddpg,
    base_dqn,
    base_dreamer_v3,
    base_go_dqn,
    base_go_explore,
    base_muzero,
    base_planet,
    base_ppo,
    base_r2d2,
    base_rainbow,
    base_sac,
    base_snd,
    base_stochastic_muzero,
    base_world_models,
)


class _CommonOption:
    def use_framework(self) -> str:
        return "torch"


class Test_agent57_light_CPU(_CommonOption, base_agent57_light.LongCase):
    pass


class Test_agent57(_CommonOption, base_agent57.LongCase):
    pass


class Test_dqn(_CommonOption, base_dqn.LongCase):
    pass


class Test_rainbow_CPU(_CommonOption, base_rainbow.LongCase):
    pass
