from dataclasses import dataclass

import numpy as np
import pytest

import srl
from srl.base.define import ObservationModes, RLBaseTypes, SpaceTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.rl.config import RLConfig
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.utils import common


class StubEnv(EnvBase):
    def __init__(self, action_space, observation_space) -> None:
        self._action_space = action_space
        self._observation_space = observation_space
        self.state = 0

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def max_episode_steps(self) -> int:
        return 1

    @property
    def player_num(self) -> int:
        return 1

    def reset(self):
        return self.state, {}

    def step(self, action):
        return self.state, [0], True, {}

    @property
    def next_player_index(self) -> int:
        return 0

    def backup(self):
        return None

    def restore(self, data) -> None:
        return


registration.register(id="config_StubEnv", entry_point=__name__ + ":StubEnv")


@dataclass
class TestConfig(RLConfig):
    set_base_action_type: RLBaseTypes = RLBaseTypes.DISCRETE
    set_base_observation_type: RLBaseTypes = RLBaseTypes.DISCRETE

    def get_name(self) -> str:
        return "test"

    def get_base_action_type(self) -> RLBaseTypes:
        return self.set_base_action_type

    def get_base_observation_type(self) -> RLBaseTypes:
        return self.set_base_observation_type

    def get_framework(self) -> str:
        return ""


_DS = DiscreteSpace(5)
_BS = BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS)
_ANY = RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS


@pytest.mark.parametrize(
    """
    env_obs_space,
    rl_base_obs_type,
    observation_mode,
    true_env_obs_space,
    true_rl_obs_space,
    """,
    [
        # Spaces
        [
            DiscreteSpace(5),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            DiscreteSpace(5),
            ArrayDiscreteSpace(1, 0, 4),
        ],
        [
            ArrayDiscreteSpace(5, 0, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ArrayDiscreteSpace(5, 0, 1),
            ArrayDiscreteSpace(5, 0, 1),
        ],
        [
            ContinuousSpace(-1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ContinuousSpace(-1, 1),
            BoxSpace((1,), -1, 1, np.float32, SpaceTypes.CONTINUOUS),
        ],
        [
            ArrayContinuousSpace(5, -1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ArrayContinuousSpace(5, -1, 1),
            BoxSpace((5,), -1, 1, np.float32, SpaceTypes.CONTINUOUS),
        ],
        [
            BoxSpace((2, 3), -1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            BoxSpace((2, 3), -1, 1),
            BoxSpace((2, 3), -1, 1, np.float32, SpaceTypes.CONTINUOUS),
        ],
        # obs RENDER_IMAGE
        [
            DiscreteSpace(5),
            RLBaseTypes.DISCRETE,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            ArrayDiscreteSpace(4 * 4 * 3, 0, 255),
        ],
        [
            DiscreteSpace(5),
            RLBaseTypes.CONTINUOUS,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            BoxSpace((4, 4, 3), 0, 255, np.float32, SpaceTypes.CONTINUOUS),
        ],
        [
            DiscreteSpace(5),
            RLBaseTypes.IMAGE,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            BoxSpace((4, 4, 3), 0, 255, np.float32, SpaceTypes.COLOR),
        ],
        # BoxSpace
        [
            BoxSpace((3,), -1, 1, stype=SpaceTypes.DISCRETE),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE,
            ObservationModes.ENV,
            BoxSpace((3,), -1, 1, np.float32, SpaceTypes.DISCRETE),
            ArrayDiscreteSpace(3, -1, 1),
        ],
        # BoxSpace image
        [
            BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE,
            ObservationModes.ENV,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.GRAY_3ch),
            BoxSpace((4, 4, 3), 0, 255, np.float32, SpaceTypes.GRAY_3ch),
        ],
    ],
)
@pytest.mark.parametrize(
    """
    env_act_space,
    rl_base_act_type,
    true_rl_act_space,
    """,
    [
        # Spaces DISCRETE
        [DiscreteSpace(5), RLBaseTypes.DISCRETE, DiscreteSpace(5)],
        [ArrayDiscreteSpace(2, 0, 1), RLBaseTypes.DISCRETE, DiscreteSpace(4)],
        [ContinuousSpace(-1, 1), RLBaseTypes.DISCRETE, DiscreteSpace(5)],
        [ArrayContinuousSpace(2, -1, 1), RLBaseTypes.DISCRETE, DiscreteSpace(25)],
        [BoxSpace((2, 1), -1, 1), RLBaseTypes.DISCRETE, DiscreteSpace(25)],
        # Spaces CONT
        [DiscreteSpace(5), RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(1, 0, 4)],
        [ArrayDiscreteSpace(2, 0, 1), RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(2, 0, 1)],
        [ContinuousSpace(-1, 1), RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(1, -1, 1)],
        [ArrayContinuousSpace(2, -1, 1), RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(2, -1, 1)],
        [BoxSpace((2, 1), -1, 1), RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(2, -1, 1)],
        # Spaces ANY
        [DiscreteSpace(5), _ANY, DiscreteSpace(5)],
        [ArrayDiscreteSpace(2, 0, 1), _ANY, DiscreteSpace(4)],
        [ContinuousSpace(-1, 1), _ANY, ArrayContinuousSpace(1, -1, 1)],
        [ArrayContinuousSpace(2, -1, 1), _ANY, ArrayContinuousSpace(2, -1, 1)],
        [BoxSpace((2, 1), -1, 1), _ANY, ArrayContinuousSpace(2, -1, 1)],
    ],
)
def test_setup(
    # obs
    env_obs_space,
    rl_base_obs_type,
    observation_mode,
    true_env_obs_space,
    true_rl_obs_space,
    # act
    env_act_space,
    rl_base_act_type,
    true_rl_act_space,
):
    common.logger_print()
    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=env_act_space,
                observation_space=env_obs_space,
            ),
        )
    )
    # base_observation_typeはANYでspaceの状態が反映される場合のみを確認
    rl_config = TestConfig(set_base_action_type=rl_base_act_type, set_base_observation_type=rl_base_obs_type)
    rl_config.observation_mode = observation_mode
    rl_config.setup(env)

    assert len(rl_config.observation_spaces_of_env) == 1
    assert true_env_obs_space == rl_config.observation_spaces_of_env[0]
    assert rl_config.observation_space == rl_config.observation_space_one_step
    assert true_rl_obs_space == rl_config.observation_space

    assert true_rl_act_space == rl_config.action_space


@pytest.mark.parametrize(
    """
    env_obs_space,
    rl_base_obs_type,
    observation_mode,
    true_env_obs_space,
    true_rl_obs_space_one_step,
    true_rl_obs_space,
    """,
    [
        # spaces
        [
            DiscreteSpace(5),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            DiscreteSpace(5),
            ArrayDiscreteSpace(1, 0, 4),
            ArrayDiscreteSpace(3, 0, 4),
        ],
        [
            ArrayDiscreteSpace(5, 0, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ArrayDiscreteSpace(5, 0, 1),
            ArrayDiscreteSpace(5, 0, 1),
            ArrayDiscreteSpace(15, 0, 1),
        ],
        [
            ContinuousSpace(-1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ContinuousSpace(-1, 1),
            BoxSpace((1,), -1, 1),
            BoxSpace((3, 1), -1, 1),
        ],
        [
            ArrayContinuousSpace(5, -1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            ArrayContinuousSpace(5, -1, 1),
            BoxSpace((5,), -1, 1),
            BoxSpace((3, 5), -1, 1),
        ],
        [
            BoxSpace((2, 5), -1, 1),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
            ObservationModes.ENV,
            BoxSpace((2, 5), -1, 1),
            BoxSpace((2, 5), -1, 1, np.float32, SpaceTypes.CONTINUOUS),
            BoxSpace((3, 2, 5), -1, 1, np.float32, SpaceTypes.CONTINUOUS),
        ],
        # obs RENDER_IMAGE
        [
            DiscreteSpace(5),
            RLBaseTypes.DISCRETE,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            ArrayDiscreteSpace(4 * 4 * 3, 0, 255),
            ArrayDiscreteSpace(3 * 4 * 4 * 3, 0, 255),
        ],
        [
            DiscreteSpace(5),
            RLBaseTypes.CONTINUOUS,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            BoxSpace((4, 4, 3), 0, 255, np.float32, SpaceTypes.CONTINUOUS),
            BoxSpace((3, 4, 4, 3), 0, 255, np.float32, SpaceTypes.CONTINUOUS),
        ],
        [
            DiscreteSpace(5),
            RLBaseTypes.IMAGE,
            ObservationModes.RENDER_IMAGE,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, SpaceTypes.COLOR),
            BoxSpace((4, 4, 3), 0, 255, np.float32, SpaceTypes.COLOR),
            BoxSpace((3, 4, 4, 3), 0, 255, np.float32, SpaceTypes.COLOR),
        ],
        # BoxSpace
        [
            BoxSpace((3,), -1, 1, stype=SpaceTypes.DISCRETE),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE,
            ObservationModes.ENV,
            BoxSpace((3,), -1, 1, np.float32, SpaceTypes.DISCRETE),
            ArrayDiscreteSpace(3, -1, 1),
            ArrayDiscreteSpace(3 * 3, -1, 1),
        ],
        # BoxSpace image
        [
            BoxSpace((4, 4), 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch),
            RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE,
            ObservationModes.ENV,
            BoxSpace((4, 4), 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch),
            BoxSpace((4, 4), 0, 255, np.float32, stype=SpaceTypes.GRAY_2ch),
            BoxSpace((4, 4, 3), 0, 255, np.float32, stype=SpaceTypes.IMAGE),
        ],
    ],
)
def test_setup_window(
    # obs
    env_obs_space,
    rl_base_obs_type,
    observation_mode,
    true_env_obs_space,
    true_rl_obs_space_one_step,
    true_rl_obs_space,
):
    common.logger_print()
    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=DiscreteSpace(5),
                observation_space=env_obs_space,
            ),
        )
    )
    rl_config = TestConfig(set_base_action_type=RLBaseTypes.DISCRETE, set_base_observation_type=rl_base_obs_type)
    rl_config.observation_mode = observation_mode
    rl_config.window_length = 3
    rl_config.setup(env)

    assert len(rl_config.observation_spaces_of_env) == 1
    assert true_env_obs_space == rl_config.observation_spaces_of_env[0]
    assert true_rl_obs_space_one_step == rl_config.observation_space_one_step
    assert true_rl_obs_space == rl_config.observation_space


def test_setup_override():
    env_obs_space = BoxSpace((2, 2, 2), 0, 1, stype=SpaceTypes.CONTINUOUS)
    override_obs_type = SpaceTypes.IMAGE
    true_rl_obs_space = BoxSpace((2, 2, 2), 0, 1, stype=SpaceTypes.IMAGE)

    env_act_space = BoxSpace((2, 2, 2), 0, 1, stype=SpaceTypes.DISCRETE)
    rl_base_act_type = RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS
    override_act_type = RLBaseTypes.CONTINUOUS
    true_rl_act_space = ArrayContinuousSpace(2 * 2 * 2, 0, 1)

    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=env_act_space,
                observation_space=env_obs_space,
            ),
        )
    )
    rl_config = TestConfig(
        set_base_action_type=rl_base_act_type,
        set_base_observation_type=RLBaseTypes.IMAGE,
    )
    rl_config.override_observation_type = override_obs_type
    rl_config.override_action_type = override_act_type
    rl_config.setup(env)

    assert true_rl_obs_space == rl_config.observation_space
    assert len(rl_config.observation_spaces_one_step) == 1
    assert true_rl_obs_space == rl_config.observation_spaces_one_step[0]

    assert true_rl_act_space == rl_config.action_space


@pytest.mark.parametrize(
    """
    window_length,
    true_rl_obs_space,
    true_rl_obs_types,
    true_rl_act_space,
    true_rl_act_types,
    """,
    [
        [
            1,
            MultiSpace(
                [
                    DiscreteSpace(5),
                    BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
                ]
            ),
            [SpaceTypes.DISCRETE, SpaceTypes.CONTINUOUS, SpaceTypes.IMAGE, SpaceTypes.IMAGE],
            MultiSpace(
                [
                    DiscreteSpace(5),
                    BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                ]
            ),
            [SpaceTypes.DISCRETE, SpaceTypes.CONTINUOUS, SpaceTypes.IMAGE],
        ],
        [
            5,
            MultiSpace(
                [
                    BoxSpace((5, 1), 0, 4, np.uint64, SpaceTypes.DISCRETE),
                    BoxSpace((5, 5), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                    BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                    BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
                ]
            ),
            [SpaceTypes.DISCRETE, SpaceTypes.CONTINUOUS, SpaceTypes.IMAGE, SpaceTypes.IMAGE],
            MultiSpace(
                [
                    DiscreteSpace(5),
                    BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                ]
            ),
            [SpaceTypes.DISCRETE, SpaceTypes.CONTINUOUS, SpaceTypes.IMAGE],
        ],
    ],
)
def test_setup_multi_space(
    window_length,
    true_rl_obs_space,
    true_rl_obs_types,
    true_rl_act_space,
    true_rl_act_types,
):
    pytest.skip("TODO")

    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=MultiSpace(
                    [
                        DiscreteSpace(5),
                        BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                        BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                    ]
                ),
                observation_space=MultiSpace(
                    [
                        DiscreteSpace(5),
                        BoxSpace((5,), 0, 1, np.float32, SpaceTypes.CONTINUOUS),
                        BoxSpace((4, 4, 3), 0, 255, np.uint8, stype=SpaceTypes.GRAY_3ch),
                    ]
                ),
            ),
        )
    )
    rl_config = TestConfig(
        set_base_action_type=RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
        set_base_observation_type=RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
    )
    rl_config.observation_mode = ObservationModes.ENV | ObservationModes.RENDER_IMAGE
    rl_config.window_length = window_length
    rl_config.setup(env)

    assert true_rl_obs_space == rl_config.observation_space
    assert len(true_rl_obs_space.spaces) == len(rl_config.observation_spaces_one_step)
    for i in range(len(rl_config.observation_spaces_one_step)):
        assert true_rl_obs_space.spaces[i] == rl_config.observation_spaces_one_step[i]
        assert true_rl_obs_types[i] == rl_config.observation_spaces_one_step[i].stype

    assert isinstance(rl_config.action_space, MultiSpace)
    assert true_rl_act_space == rl_config.action_space
    for i in range(len(true_rl_act_space.spaces)):
        assert true_rl_act_space.spaces[i] == rl_config.action_space.spaces[i]
        assert true_rl_act_types[i] == rl_config.action_space.spaces[i].stype


def test_copy():
    config = TestConfig()
    config.window_length = 4

    assert not config.is_setup
    config.setup(srl.make_env("Grid"))
    config.window_length = 2
    assert config.is_setup
    config.setup(srl.make_env("Grid"))
    assert config.is_setup

    config2 = config.copy()
    assert config.is_setup
    assert config.window_length == 2
    assert config2.name == "test"
