from tests.algorithms_ import (
    base_agent57,
    base_agent57_light,
    base_alphazero,
    base_c51,
    base_ddpg,
    base_dqn,
    base_dqn_not,
    base_dreamer_v3,
    base_go_dqn,
    base_go_explore,
    base_godq_v1,
    base_godq_v1_lstm,
    base_muzero,
    base_planet,
    base_ppo,
    base_ppo_v,
    base_r2d2,
    base_rainbow,
    base_sac,
    base_sac_not,
    base_snd,
    base_stochastic_muzero,
    base_world_models,
)


class _CommonOption:
    def use_framework(self) -> str:
        return "torch"


class Test_agent57_light(_CommonOption, base_agent57_light.LongCase):
    pass


class Test_agent57(_CommonOption, base_agent57.LongCase):
    pass


class Test_dqn(_CommonOption, base_dqn.LongCase):
    pass


class Test_rainbow(_CommonOption, base_rainbow.LongCase):
    pass


class Test_dqn_not(_CommonOption, base_dqn_not.LongCase):
    pass


class Test_godq_v1(_CommonOption, base_godq_v1.LongCase):
    pass


class Test_godq_v1_lstm(_CommonOption, base_godq_v1_lstm.LongCase):
    pass


class Test_ppo_v(_CommonOption, base_ppo_v.LongCase):
    pass


class Test_sac_not(_CommonOption, base_sac_not.LongCase):
    pass
