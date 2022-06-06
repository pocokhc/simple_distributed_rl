import itertools
import unittest

import numpy as np
from srl.base.env.spaces import BoxSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def _check_action(self, decode_action, true_shape, true_action):
        self.assertTrue(isinstance(decode_action, np.ndarray))
        self.assertTrue("float" in str(decode_action.dtype))
        self.assertTrue(decode_action.shape == true_shape)
        if true_action is not None:
            np.testing.assert_array_equal(decode_action, true_action)

    def test_space1(self):
        self.space = BoxSpace((1,), -1, 3)
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, (1,), None)
            self.assertTrue(np.min(action) >= -1)
            self.assertTrue(np.max(action) <= 3)

        # action discrete
        self.space.set_action_division(5)
        true_tbl = [
            [-1],
            [0],
            [1],
            [2],
            [3],
        ]
        np.testing.assert_array_equal(true_tbl[0], self.space.action_tbl[0])
        decode_action = self.tester.check_action_discrete(
            true_n=5,
            action=3,
        )
        self._check_action(decode_action, (1,), [2])

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=1,
            true_low=[-1],
            true_high=[3],
            action=[1.1],
        )
        self._check_action(decode_action, (1,), [1.1])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(1,),
            true_low=[-1],
            true_high=[3],
            state=np.array([1.1]),
            encode_state=[1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(1,),
            true_low=[-1],
            true_high=[3],
            state=np.array([1.1]),
            encode_state=[1.1],
        )

        # eq
        self.assertTrue(self.space == BoxSpace((1,), -1, 3))
        self.assertTrue(self.space != BoxSpace((1,), -1, 2))

    def test_space2(self):
        self.space = BoxSpace((3, 2), -1, 3)
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, (3, 2), None)
            self.assertTrue(np.min(action) >= -1)
            self.assertTrue(np.max(action) <= 3)

        # action discrete
        self.space.set_action_division(5)
        _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
        true_tbl = list(itertools.product(_t, _t, _t))
        for a in range(len(true_tbl)):
            np.testing.assert_array_equal(true_tbl[a], self.space.action_tbl[a])
        decode_action = self.tester.check_action_discrete(
            true_n=5 ** (3 * 2),
            action=3,
        )
        self._check_action(decode_action, (3, 2), true_tbl[3])

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=6,
            true_low=[-1] * 6,
            true_high=[3] * 6,
            action=[1.1, 2.2, 3.0, -1.0, 1.5, 1.6],
        )
        self._check_action(decode_action, (3, 2), [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3, 2),
            true_low=np.full((3, 2), -1).tolist(),
            true_high=np.full((3, 2), 3).tolist(),
            state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]]),
            encode_state=[[1, 2], [3, -1], [1, 2]],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3, 2),
            true_low=np.full((3, 2), -1).tolist(),
            true_high=np.full((3, 2), 3).tolist(),
            state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]]),
            encode_state=[[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]],
        )

        # eq
        self.assertTrue(self.space == BoxSpace((3, 2), -1, 3))
        self.assertTrue(self.space != BoxSpace((3, 3), -1, 3))

    def test_inf(self):
        self.space = BoxSpace((3, 2))
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, (3, 2), None)

        # action discrete
        self.space.set_action_division(5)
        decode_action = self.tester.check_action_discrete(
            true_n=0,
            action=3,
        )
        self._check_action(decode_action, (3, 2), np.full((3, 2), 3))

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=6,
            true_low=[-np.inf] * 6,
            true_high=[np.inf] * 6,
            action=[1.1, 2.2, 3.0, -1.0, 1.5, 1.6],
        )
        self._check_action(decode_action, (3, 2), [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3, 2),
            true_low=np.full((3, 2), -np.inf).tolist(),
            true_high=np.full((3, 2), np.inf).tolist(),
            state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]]),
            encode_state=[[1, 2], [3, -1], [1, 2]],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3, 2),
            true_low=np.full((3, 2), -np.inf).tolist(),
            true_high=np.full((3, 2), np.inf).tolist(),
            state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]]),
            encode_state=[[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]],
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_inf", verbosity=2)
