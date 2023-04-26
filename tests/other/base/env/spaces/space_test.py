from typing import Any, List, Tuple

import numpy as np

from srl.base.env.base import SpaceBase


class SpaceTest:
    def __init__(self, space: SpaceBase):
        self.space = space

    def check_action_discrete(self, true_n: int, action: int) -> Any:
        n = self.space.get_action_discrete_info()
        assert n == true_n
        return self.space.action_discrete_decode(action)

    def check_action_encode(self, action: Any, encode_action: int):
        assert self.space.action_discrete_encode(action) == encode_action

    def check_action_continuous(
        self,
        true_n: int,
        true_low: List[float],
        true_high: List[float],
        action: List[float],
    ) -> Any:
        n, low, high = self.space.get_action_continuous_info()
        assert n == true_n
        assert isinstance(low, np.ndarray)
        np.testing.assert_array_equal(low, true_low)
        assert isinstance(high, np.ndarray)
        np.testing.assert_array_equal(high, true_high)

        return self.space.action_continuous_decode(action)

    def check_observation_discrete(
        self,
        true_shape: Tuple[int, ...],
        state,
        encode_state,
    ):
        shape = self.space.observation_shape
        assert isinstance(shape, tuple)
        assert shape == true_shape

        state = self.space.observation_discrete_encode(state)
        assert isinstance(state, np.ndarray)
        assert "int" in str(state.dtype)
        np.testing.assert_array_equal(state, encode_state)

    def check_observation_continuous(
        self,
        true_shape: Tuple[int, ...],
        state,
        encode_state,
    ):
        shape = self.space.observation_shape
        assert isinstance(shape, tuple)
        assert shape == true_shape

        state = self.space.observation_continuous_encode(state)
        assert isinstance(state, np.ndarray)
        assert "float" in str(state.dtype)
        np.testing.assert_array_equal(state, encode_state)
