import numpy as np
import pytest

from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.processors.normalize_processor import NormalizeProcessor


@pytest.mark.parametrize(
    "space, true_space, state, true_state",
    [
        [DiscreteSpace(5), ContinuousSpace(-1, 2), 4, 2.0],
        [ContinuousSpace(1, 4), ContinuousSpace(-1, 2), 4, 2],
        [ArrayDiscreteSpace(2, 1, 4), ArrayContinuousSpace(2, -1, 2), [1, 4], np.array([-1, 2], np.float32)],
        [ArrayContinuousSpace(2, 1, 4), ArrayContinuousSpace(2, -1, 2), [1, 4], np.array([-1, 2], np.float32)],
        [
            BoxSpace((2, 1), 1, 4),
            BoxSpace((2, 1), -1, 2),
            np.array([[1], [4]], np.float32),
            np.array([[-1], [2]], np.float32),
        ],
    ],
)
def test_1(space, true_space, state, true_state):
    processor = NormalizeProcessor((-1, 2))

    # --- change space
    new_space = processor.remap_observation_space(space)
    print(new_space)
    print(true_space)
    assert new_space == true_space

    # --- decode
    new_state = processor.remap_observation(state, space, new_space)
    print(new_state)
    if isinstance(true_state, np.ndarray):
        assert (new_state == true_state).all()
    else:
        assert new_state == true_state
