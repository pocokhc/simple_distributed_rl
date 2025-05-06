from tests.algorithms_ import (
    base_agent57,
    base_agent57_light,
    base_alphazero,
    base_c51,
    base_ddpg,
    base_diamond,
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
        return "tensorflow"


class Test_agent57_light(_CommonOption, base_agent57_light.LongCase):
    pass


class Test_agent57(_CommonOption, base_agent57.LongCase):
    pass


class Test_alphazero(_CommonOption, base_alphazero.LongCase):
    pass


class Test_c51(_CommonOption, base_c51.LongCase):
    pass


class Test_ddpg(_CommonOption, base_ddpg.LongCase):
    pass


class Test_dqn(_CommonOption, base_dqn.LongCase):
    pass


class Test_dreamer_v3(_CommonOption, base_dreamer_v3.LongCase):
    pass


class Test_diamond(_CommonOption, base_diamond.LongCase):
    pass


class Test_muzero(_CommonOption, base_muzero.LongCase):
    pass


class Test_planet(_CommonOption, base_planet.LongCase):
    pass


class Test_go_explore(_CommonOption, base_go_explore.LongCase):
    pass


class Test_go_dqn(_CommonOption, base_go_dqn.LongCase):
    pass


class Test_ppo(_CommonOption, base_ppo.LongCase):
    pass


class Test_r2d2(_CommonOption, base_r2d2.LongCase):
    pass


class Test_rainbow(_CommonOption, base_rainbow.LongCase):
    pass


class Test_sac(_CommonOption, base_sac.LongCase):
    pass


class Test_snd(_CommonOption, base_snd.LongCase):
    pass


class Test_stochastic_muzero(_CommonOption, base_stochastic_muzero.LongCase):
    pass


class Test_world_models(_CommonOption, base_world_models.LongCase):
    pass
