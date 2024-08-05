from tests.algorithms_ import (
    base_dynaq,
    base_go_dynaq,
    base_mcts,
    base_ql,
    base_ql_agent57,
    base_search_dynaq,
    base_vanilla_policy,
)


class _CommonOption:
    def use_framework(self) -> str:
        return ""

    def get_framework(self) -> str:
        return ""

    def get_device(self) -> str:
        return ""


class Test_dynaq_base(_CommonOption, base_dynaq.BaseCase):
    pass


class Test_mcts_base(_CommonOption, base_mcts.BaseCase):
    pass


class Test_ql_base(_CommonOption, base_ql.BaseCase):
    pass


class Test_ql_agent57_base(_CommonOption, base_ql_agent57.BaseCase):
    pass


class Test_search_dynaq_base(_CommonOption, base_search_dynaq.BaseCase):
    pass


class Test_go_dynaq_base(_CommonOption, base_go_dynaq.BaseCase):
    pass


class Test_vanilla_policy_base(_CommonOption, base_vanilla_policy.BaseCase):
    pass
