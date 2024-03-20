import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.rl.functions import helper


# -----------------------------------
# test get_random_max_index
# -----------------------------------
@pytest.mark.parametrize(
    "arr",
    [
        [1.0, 0.9, 1.1, 0.8],
        np.array([1.0, 0.9, 1.1, 0.8], dtype=float),
        np.array([1, 0, 2, -1], dtype=int),
    ],
)
@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [0, 1],
    ],
)
def test_get_random_max_index_1(arr, invalid_actions):
    assert helper.get_random_max_index(arr, invalid_actions) == 2


@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [0],
    ],
)
def test_get_random_max_index_2(invalid_actions):
    arr = [1.0, 0.9, 1.1, 1.1, 1.1, 0.5]
    for _ in range(100):
        n = helper.get_random_max_index(arr, invalid_actions)
        assert n in [2, 3, 4]


@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [i for i in range(0, 50)],
    ],
)
def test_get_random_max_index_3(invalid_actions):
    arr = [0.5] * 100
    arr.append(0.6)
    assert helper.get_random_max_index(arr, invalid_actions) == 100


def test_get_random_max_index_4():
    arr = [0, 1.1, 1.2, 1.3, 1.1]
    assert helper.get_random_max_index(arr, [3]) == 2


def test_create_fancy_index_for_invalid_actions():
    idx_list = [
        [1, 2, 5],
        [2],
        [2, 3],
    ]
    idx1, idx2 = helper.create_fancy_index_for_invalid_actions(idx_list)
    print(idx1)
    print(idx2)
    assert idx1 == [0, 0, 0, 1, 2, 2]
    assert idx2 == [1, 2, 5, 2, 2, 3]
