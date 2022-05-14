import itertools
import unittest

import numpy as np
from srl.base.env.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace


class Test(unittest.TestCase):
    def test_discrete(self):
        space = DiscreteSpace(5)
        self.assertTrue(space.n == 5)

        # sample
        actions = [space.sample([3]) for _ in range(100)]
        actions = sorted(list(set(actions)))
        np.testing.assert_array_equal(actions, [0, 1, 2, 4])

        # action_discrete
        self.assertTrue(space.get_action_discrete_info() == 5)
        self.assertTrue(space.action_discrete_encode(2) == 2)
        self.assertTrue(space.action_discrete_decode(3) == 3)

        # action_continuous
        n, low, high = space.get_action_continuous_info()
        self.assertTrue(n == 1)
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [4])
        self.assertTrue(space.action_continuous_encode(2) == [2.0])
        self.assertTrue(space.action_continuous_decode([3.3]) == 3)

        # observation discrete
        shape, low, high = space.get_observation_discrete_info()
        self.assertTrue(shape == (1,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [4])
        self.assertTrue(space.observation_discrete_encode(2) == [2.0])

        # observation continuous
        shape, low, high = space.get_observation_continuous_info()
        self.assertTrue(shape == (1,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [4])
        self.assertTrue(space.observation_continuous_encode(2) == [2.0])

    def test_array_discrete(self):
        space = ArrayDiscreteSpace([2, 5, 3])
        np.testing.assert_array_equal(space.nvec, [2, 5, 3])

        # sample
        for _ in range(100):
            action = space.sample()
            self.assertTrue(0 <= action[0] <= 1)
            self.assertTrue(0 <= action[1] <= 5)
            self.assertTrue(0 <= action[2] <= 3)

        # action_discrete
        n = space.get_action_discrete_info()
        self.assertTrue(n == 2 * 5 * 3)
        self.assertTrue(space.action_discrete_encode([0, 0, 1]) == 1)
        np.testing.assert_array_equal(space.action_discrete_decode(1), [0, 0, 1])

        # action_continuous
        n, low, high = space.get_action_continuous_info()
        self.assertTrue(n == 3)
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0, 0, 0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [1, 4, 2])
        self.assertTrue(space.action_continuous_encode([0, 0, 1]) == [0.0, 0.0, 1.0])
        self.assertTrue(space.action_continuous_decode([0.1, 0.6, 0.9]) == [0, 1, 1])

        # observation discrete
        shape, low, high = space.get_observation_discrete_info()
        self.assertTrue(shape == (3,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0, 0, 0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [1, 4, 2])
        np.testing.assert_array_equal(space.observation_discrete_encode([0, 0, 1]), np.array([0, 0, 1]))

        # observation continuous
        shape, low, high = space.get_observation_continuous_info()
        self.assertTrue(shape == (3,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, [0, 0, 0])
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, [1, 4, 2])
        np.testing.assert_array_equal(space.observation_continuous_encode([0, 0, 1]), np.array([0.0, 0.0, 1.0]))

    def test_box1(self):
        space = BoxSpace(low=-1, high=3, shape=(1,))

        # --- sample
        actions = [space.sample() for _ in range(100)]
        self.assertTrue(np.min(actions) >= -1)
        self.assertTrue(np.max(actions) <= 3)

        # --- action_discrete
        space.set_division(5)
        self.assertTrue(space.get_action_discrete_info() == 5)
        true_tbl = [
            [-1],
            [0],
            [1],
            [2],
            [3],
        ]
        np.testing.assert_array_equal(true_tbl, space.action_tbl)
        np.testing.assert_array_equal(space.action_discrete_decode(3), [2])

        # --- action_continuous
        n, low, high = space.get_action_continuous_info()
        self.assertTrue(n == 1)
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((1,), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((1,), 3))
        dat = np.array([1.1])
        en_dat = [1.1]
        np.testing.assert_array_equal(space.action_continuous_encode(dat), en_dat)
        np.testing.assert_array_equal(space.action_continuous_decode(en_dat), dat)

        # --- observation discrete
        shape, low, high = space.get_observation_discrete_info()
        self.assertTrue(shape == (1,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((1,), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((1,), 3))
        np.testing.assert_array_equal(space.observation_discrete_encode(np.array([1.1])), [1])

        # --- observation continuous
        shape, low, high = space.get_observation_continuous_info()
        self.assertTrue(shape == (1,))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((1,), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((1,), 3))
        np.testing.assert_array_equal(space.observation_continuous_encode(np.array([1.1])), [1.1])

    def test_box2(self):
        space = BoxSpace(low=-1, high=3, shape=(3, 2))

        # --- sample
        actions = [space.sample() for _ in range(100)]
        self.assertTrue(np.min(actions) >= -1)
        self.assertTrue(np.max(actions) <= 3)

        # --- to descrete
        space.set_division(5)
        n = space.get_action_discrete_info()
        self.assertTrue(n == 5 ** (3 * 2))
        _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
        true_tbl = list(itertools.product(_t, _t, _t))
        for a in range(len(true_tbl)):
            np.testing.assert_array_equal(true_tbl[a], space.action_tbl[a])
        np.testing.assert_array_equal(space.action_discrete_decode(3), true_tbl[3])

        # --- action_continuous
        n, low, high = space.get_action_continuous_info()
        self.assertTrue(n == 6)
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((3, 2), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((3, 2), 3))
        dat = np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])
        en_dat = [1.1, 2.2, 3.0, -1.0, 1.5, 1.6]
        np.testing.assert_array_equal(space.action_continuous_encode(dat), en_dat)
        np.testing.assert_array_equal(space.action_continuous_decode(en_dat), dat)

        # --- observation discrete
        shape, low, high = space.get_observation_discrete_info()
        self.assertTrue(shape == (3, 2))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((3, 2), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((3, 2), 3))
        dat = np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]])
        en_dat = np.array([[1, 2], [3, -1], [1, 2]])
        np.testing.assert_array_equal(space.observation_discrete_encode(dat), en_dat)

        # --- observation continuous
        shape, low, high = space.get_observation_continuous_info()
        self.assertTrue(shape == (3, 2))
        self.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, np.full((3, 2), -1))
        self.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, np.full((3, 2), 3))
        dat = np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]])
        np.testing.assert_array_equal(space.observation_continuous_encode(dat), dat)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_box2", verbosity=2)
