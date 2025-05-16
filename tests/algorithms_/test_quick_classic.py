from typing import Tuple

import pytest

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_quick_case import CommonQuickCase


class _CommonOption:
    def use_framework(self) -> str:
        return ""


# --- QL base


class Test_ql(_CommonOption, CommonQuickCase):
    @pytest.fixture(params=["", "random", "normal"])
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import ql

        return ql.Config(q_init=rl_param), {}


class Test_ql_agent57(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import ql_agent57

        return ql_agent57.Config(), {}


class Test_dynaq(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import dynaq

        return dynaq.Config(), {}


class Test_search_dynaq(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import search_dynaq

        return search_dynaq.Config(), {}


class Test_go_dynaq(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import go_dynaq

        return go_dynaq.Config(), {}


# --- MCTS base


class Test_mcts(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import mcts

        return mcts.Config(num_simulations=1), dict(use_layer_processor=True)


# --- Policy base


class Test_vanilla_policy_dis(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import vanilla_policy

        return vanilla_policy.Config(), {}


class Test_vanilla_policy_con(_CommonOption, CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import vanilla_policy

        rl_config = vanilla_policy.Config()
        rl_config.override_action_type = RLBaseTypes.CONTINUOUS
        return rl_config, {}
