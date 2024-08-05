from tests.algorithms_ import (
    base_agent57,
    base_agent57_light,
    base_alphazero,
    base_c51,
    base_ddpg,
    base_dqn,
    base_dreamer_v3,
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
        return "tensorflow"


class Test_agent57_light(_CommonOption, base_agent57_light.QuickCase):
    pass


class Test_agent57(_CommonOption, base_agent57.QuickCase):
    pass


class Test_alphazero(_CommonOption, base_alphazero.QuickCase):
    pass


class Test_c51(_CommonOption, base_c51.QuickCase):
    pass


class Test_ddpg(_CommonOption, base_ddpg.QuickCase):
    pass


class Test_dqn(_CommonOption, base_dqn.QuickCase):
    pass


class Test_dreamer_v3(_CommonOption, base_dreamer_v3.QuickCase):
    pass


class Test_muzero(_CommonOption, base_muzero.QuickCase):
    pass


class Test_planet(_CommonOption, base_planet.QuickCase):
    pass


class Test_go_explore(_CommonOption, base_go_explore.QuickCase):
    pass


class Test_ppo(_CommonOption, base_ppo.QuickCase):
    pass


class Test_r2d2(_CommonOption, base_r2d2.QuickCase):
    pass


class Test_rainbow(_CommonOption, base_rainbow.QuickCase):
    pass


class Test_sac(_CommonOption, base_sac.QuickCase):
    pass


class Test_snd(_CommonOption, base_snd.QuickCase):
    pass


class Test_stochastic_muzero(_CommonOption, base_stochastic_muzero.QuickCase):
    pass


class Test_world_models(_CommonOption, base_world_models.QuickCase):
    pass
