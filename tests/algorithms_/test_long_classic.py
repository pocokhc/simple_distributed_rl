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


# --- QL base


class Test_ql(_CommonOption, base_ql.LongCase):
    pass


class Test_ql_agent57(_CommonOption, base_ql_agent57.LongCase):
    pass


class Test_dynaq(_CommonOption, base_dynaq.LongCase):
    pass


class Test_search_dynaq(_CommonOption, base_search_dynaq.LongCase):
    pass


class Test_go_dynaq(_CommonOption, base_go_dynaq.LongCase):
    pass


# --- MCTS base


class Test_mcts(_CommonOption, base_mcts.LongCase):
    pass


# --- Policy base


class Test_vanilla_policy(_CommonOption, base_vanilla_policy.LongCase):
    pass
