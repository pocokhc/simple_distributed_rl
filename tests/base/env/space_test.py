from typing import Any, List, Tuple

import numpy as np
from srl.base.env.base import SpaceBase


class SpaceTest:
    def __init__(self, tester, space: SpaceBase):
        self.tester = tester
        self.space = space

    def check_action_discrete(self, true_n: int, action: int) -> Any:
        n = self.space.get_action_discrete_info()
        self.tester.assertTrue(n == true_n)
        return self.space.action_discrete_decode(action)

    def check_action_encode(self, action: Any, encode_action: int):
        self.tester.assertTrue(self.space.action_discrete_encode(action) == encode_action)

    def check_action_continuous(
        self,
        true_n: int,
        true_low: List[float],
        true_high: List[float],
        action: List[float],
    ) -> Any:
        n, low, high = self.space.get_action_continuous_info()
        self.tester.assertTrue(n == true_n)
        self.tester.assertTrue(isinstance(low, np.ndarray))
        np.testing.assert_array_equal(low, true_low)
        self.tester.assertTrue(isinstance(high, np.ndarray))
        np.testing.assert_array_equal(high, true_high)

        return self.space.action_continuous_decode(action)

    def check_observation_discrete(
        self,
        true_shape: Tuple[int, ...],
        state,
        encode_state: List,
    ):
        shape = self.space.observation_shape
        self.tester.assertTrue(isinstance(shape, tuple))
        self.tester.assertTrue(shape == true_shape)

        state = self.space.observation_discrete_encode(state)
        self.tester.assertTrue(isinstance(state, np.ndarray))
        self.tester.assertTrue("int" in str(state.dtype))
        np.testing.assert_array_equal(state, encode_state)

    def check_observation_continuous(
        self,
        true_shape: Tuple[int, ...],
        state,
        encode_state: List,
    ):
        shape = self.space.observation_shape
        self.tester.assertTrue(isinstance(shape, tuple))
        self.tester.assertTrue(shape == true_shape)

        state = self.space.observation_continuous_encode(state)
        self.tester.assertTrue(isinstance(state, np.ndarray))
        self.tester.assertTrue("float" in str(state.dtype))
        np.testing.assert_array_equal(state, encode_state)
