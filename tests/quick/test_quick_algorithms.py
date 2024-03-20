from tests.algorithms_ import base_dynaq, base_mcts, base_ql, base_ql_agent57, base_search_dynaq, base_vanilla_policy


class _CommonOption:
    def use_framework(self) -> str:
        return ""

    def get_framework(self) -> str:
        return ""

    def get_device(self) -> str:
        return ""


class Test_dynaq_quick(_CommonOption, base_dynaq.QuickCase):
    pass


class Test_mcts_quick(_CommonOption, base_mcts.QuickCase):
    pass


class Test_ql_quick(_CommonOption, base_ql.QuickCase):
    pass


class Test_ql_agent57_quick(_CommonOption, base_ql_agent57.QuickCase):
    pass


class Test_search_dynaq_quick(_CommonOption, base_search_dynaq.QuickCase):
    pass


class Test_vanilla_policy_dis_quick(_CommonOption, base_vanilla_policy.QuickCase_dis):
    pass


class Test_vanilla_policy_con_quick(_CommonOption, base_vanilla_policy.QuickCase_con):
    pass
