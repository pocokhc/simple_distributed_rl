import unittest

from srl.test import TestRL

try:
    from srl.algorithms import (
        agent57,
        agent57_light,
        agent57_stateful,
        alphazero,
        c51,
        ddpg,
        dqn,
        dreamer,
        dynaq,
        mcts,
        muzero,
        planet,
        ql,
        ql_agent57,
        r2d2,
        r2d2_stateful,
        rainbow,
        sac,
        search_dynaq,
        stochastic_muzero,
        vanilla_policy_continuous,
        vanilla_policy_discrete,
        world_models,
    )
except ModuleNotFoundError as e:
    print(e)


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.rl_list = [
            agent57,
            agent57_light,
            agent57_stateful,
            alphazero,
            c51,
            ddpg,
            dqn,
            # dqn_torch,
            dreamer,
            dynaq,
            mcts,
            muzero,
            planet,
            ql,
            ql_agent57,
            r2d2,
            r2d2_stateful,
            rainbow,
            sac,
            search_dynaq,
            stochastic_muzero,
            vanilla_policy_continuous,
            vanilla_policy_discrete,
            world_models,
        ]

    def _enable_image(self, rl_config):
        if rl_config.getName() in [
            "MuZero",
            "StochasticMuZero",
            "Dreamer",
            "PlaNet",
        ]:
            return True
        return False

    def test_simple_check(self):
        tester = TestRL()
        for rl_pkg in self.rl_list:
            rl_config = rl_pkg.Config()
            with self.subTest(rl_config.getName()):
                tester.simple_check(
                    rl_config,
                    enable_image=self._enable_image(rl_config),
                    check_render=False,
                )

    def test_simple_check_mp(self):
        tester = TestRL()
        for rl_pkg in self.rl_list:
            rl_config = rl_pkg.Config()
            with self.subTest(rl_config.getName()):
                tester.simple_check(
                    rl_config,
                    enable_image=self._enable_image(rl_config),
                    check_render=False,
                    is_mp=True,
                )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_simple_check", verbosity=2)
