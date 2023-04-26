import unittest


from srl.test import TestRL
from srl.utils import common

common.logger_print()


class Test_dynaq(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import dynaq

        self.rl_config = dynaq.Config()


class Test_mcts(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import mcts

        self.rl_config = mcts.Config()
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
            train_kwargs=dict(max_steps=10),
        )


class Test_ql_agent57(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import ql_agent57

        self.rl_config = ql_agent57.Config()


class Test_ql(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import ql

        self.rl_config = ql.Config()


class Test_search_dynaq(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import search_dynaq

        self.rl_config = search_dynaq.Config()


class Test_vanilla_policy(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import vanilla_policy

        self.rl_config = vanilla_policy.Config()
