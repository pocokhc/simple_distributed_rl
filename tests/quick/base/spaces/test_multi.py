import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace


def test_space_discrete():
    pytest.skip("TODO")
    space = MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )
    print(space)
    assert space.stype == SpaceTypes.MULTI

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3

    # --- eq
    assert space == MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )
    assert space != MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 1, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )


def test_sanitize():
    pytest.skip("TODO")
    space = MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )

    val = space.sanitize([1.2, [1.1, 1.2], [[1], [1]]])
    print(val)
    assert len(val) == 3
    assert space.check_val(val)
    assert val[0] == 1
    assert val[1] == [1, 1]
    np.testing.assert_array_equal(val[2], [[0], [0]])


def test_sample_discrete():
    pytest.skip("TODO")
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            BoxSpace((1, 1), -1, 0, dtype=np.int64),
        ]
    )
    for _ in range(100):
        r = space.sample(
            [
                [0, [0], np.array([[0]], dtype=np.int64)],
                [0, [1], np.array([[0]], dtype=np.int64)],
                [0, [0], np.array([[-1]], dtype=np.int64)],
                [0, [1], np.array([[-1]], dtype=np.int64)],
                [1, [0], np.array([[0]], dtype=np.int64)],
                [1, [1], np.array([[0]], dtype=np.int64)],
                [1, [0], np.array([[-1]], dtype=np.int64)],
            ]
        )
        assert len(r) == 3
        assert r[0] == 1
        assert r[1] == [1]
        assert (r[2] == [[-1]]).all()


def test_valid_actions():
    pytest.skip("TODO")
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            BoxSpace((1, 1), -1, 0, dtype=np.int64),
        ]
    )
    acts = space.get_valid_actions(
        [
            [0, [0], np.array([[0]], dtype=np.int64)],
            [0, [1], np.array([[0]], dtype=np.int64)],
            [0, [0], np.array([[-1]], dtype=np.int64)],
            [0, [1], np.array([[-1]], dtype=np.int64)],
            [1, [0], np.array([[0]], dtype=np.int64)],
            [1, [1], np.array([[0]], dtype=np.int64)],
            [1, [0], np.array([[-1]], dtype=np.int64)],
        ]
    )
    print(acts)
    assert len(acts) == 1
    assert len(acts[0]) == 3
    assert acts[0][0] == 1
    assert acts[0][1] == [1]
    assert (acts[0][2] == [[-1]]).all()
