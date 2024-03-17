from dataclasses import dataclass

import numpy as np
import pytest

import srl
from srl.base.define import EnvTypes, ObservationModes, RLBaseTypes, RLTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.rl.config import RLConfig
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase


class StubEnv(EnvBase):
    def __init__(
        self,
        action_space,
        observation_space,
    ) -> None:
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
    set_base_action_type: RLBaseTypes = RLBaseTypes.ANY
    set_base_observation_type: RLBaseTypes = RLBaseTypes.ANY

    def get_name(self) -> str:
        return "test"

    def get_base_action_type(self) -> RLBaseTypes:
        return self.set_base_action_type

    def get_base_observation_type(self) -> RLBaseTypes:
        return self.set_base_observation_type

    def get_use_framework(self) -> str:
        return ""


_DS = DiscreteSpace(5)
_BS = BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS)


@pytest.mark.parametrize(
    """
    env_obs_space,
    override_obs_type,
    rl_base_obs_type,
    observation_mode,
    window_length,
    true_rl_obs_space,
    true_rl_obs_type,
    """,
    [
        # Disc(5)
        [
            DiscreteSpace(5),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            1,
            DiscreteSpace(5),
            RLTypes.DISCRETE,
        ],
        [
            DiscreteSpace(5),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            4,
            BoxSpace((4, 1), 0, 4, np.uint64, EnvTypes.DISCRETE),
            RLTypes.DISCRETE,
        ],
        # Disc(5) -> override CONT
        [
            DiscreteSpace(5),
            EnvTypes.CONTINUOUS,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            1,
            DiscreteSpace(5),
            RLTypes.CONTINUOUS,
        ],
        [
            DiscreteSpace(5),
            EnvTypes.CONTINUOUS,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            4,
            BoxSpace((4, 1), 0, 4, np.uint64, EnvTypes.CONTINUOUS),
            RLTypes.CONTINUOUS,
        ],
        # obs RENDER_IMAGE
        [
            DiscreteSpace(5),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.RENDER_IMAGE,
            1,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, EnvTypes.COLOR),
            RLTypes.IMAGE,
        ],
        [
            DiscreteSpace(5),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.RENDER_IMAGE,
            5,
            BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, EnvTypes.COLOR),
            RLTypes.IMAGE,
        ],
        # obs RENDER_TEXT
        # [
        #    DiscreteSpace(5),
        #    EnvTypes.UNKNOWN,
        #    RLBaseTypes.ANY,
        #    ObservationModes.RENDER_TERMINAL,
        #    1,
        #    BoxSpace((4, 4, 3), 0, 255, np.uint8, EnvTypes.COLOR),
        #    RLTypes.IMAGE,
        # ],
        # [
        #    DiscreteSpace(5),
        #    EnvTypes.UNKNOWN,
        #    RLBaseTypes.ANY,
        #    ObservationModes.RENDER_TERMINAL,
        #    5,
        #    BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, EnvTypes.COLOR),
        #    RLTypes.IMAGE,
        # ],
        # BoxSpace
        [
            BoxSpace((3,), -1, 1, type=EnvTypes.DISCRETE),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            1,
            BoxSpace((3,), -1, 1, np.float32, EnvTypes.DISCRETE),
            RLTypes.DISCRETE,
        ],
        [
            BoxSpace((3,), -1, 1, type=EnvTypes.DISCRETE),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            4,
            BoxSpace((4, 3), -1, 1, np.float32, EnvTypes.DISCRETE),
            RLTypes.DISCRETE,
        ],
        # BoxSpace image
        [
            BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            1,
            BoxSpace((4, 4, 3), 0, 255, np.uint8, EnvTypes.GRAY_3ch),
            RLTypes.IMAGE,
        ],
        [
            BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
            EnvTypes.UNKNOWN,
            RLBaseTypes.ANY,
            ObservationModes.ENV,
            2,
            BoxSpace((2, 4, 4, 3), 0, 255, np.uint8, EnvTypes.GRAY_3ch),
            RLTypes.IMAGE,
        ],
        # division
        [
            BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
            EnvTypes.UNKNOWN,
            RLBaseTypes.DISCRETE,
            ObservationModes.ENV,
            1,
            BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
            RLTypes.DISCRETE,
        ],
    ],
)
@pytest.mark.parametrize(
    """
    env_act_space,
    rl_base_act_type,
    override_act_type,
    true_rl_act_space,
    true_rl_act_type,
    """,
    [
        [_DS, RLBaseTypes.DISCRETE, RLTypes.UNKNOWN, _DS, RLTypes.DISCRETE],
        [_DS, RLBaseTypes.CONTINUOUS, RLTypes.UNKNOWN, _DS, RLTypes.CONTINUOUS],
        [_DS, RLBaseTypes.ANY, RLTypes.DISCRETE, _DS, RLTypes.DISCRETE],
        [_DS, RLBaseTypes.ANY, RLTypes.CONTINUOUS, _DS, RLTypes.CONTINUOUS],
        [_DS, RLBaseTypes.ANY, RLTypes.UNKNOWN, _DS, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.DISCRETE, RLTypes.UNKNOWN, _BS, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.CONTINUOUS, RLTypes.UNKNOWN, _BS, RLTypes.CONTINUOUS],
        [_BS, RLBaseTypes.ANY, RLTypes.DISCRETE, _BS, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.ANY, RLTypes.CONTINUOUS, _BS, RLTypes.CONTINUOUS],
        [_BS, RLBaseTypes.ANY, RLTypes.UNKNOWN, _BS, RLTypes.CONTINUOUS],
    ],
)
def test_setup(
    # obs
    env_obs_space,
    override_obs_type,
    rl_base_obs_type,
    observation_mode,
    window_length,
    true_rl_obs_space,
    true_rl_obs_type,
    # act
    env_act_space,
    rl_base_act_type,
    override_act_type,
    true_rl_act_space,
    true_rl_act_type,
):
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
    rl_config.override_env_observation_type = override_obs_type
    rl_config.override_action_type = override_act_type
    rl_config.observation_mode = observation_mode
    rl_config.window_length = window_length
    rl_config.setup(env)

    assert true_rl_obs_space == rl_config.observation_space
    assert true_rl_obs_type == rl_config.observation_type
    assert len(rl_config.observation_spaces_one_step) == 1
    assert true_rl_obs_space == rl_config.observation_spaces_one_step[0]

    assert true_rl_act_space == rl_config.action_space
    assert true_rl_act_type == rl_config.action_type


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
                    BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.COLOR),
                ]
            ),
            [RLTypes.DISCRETE, RLTypes.CONTINUOUS, RLTypes.IMAGE, RLTypes.IMAGE],
            MultiSpace(
                [
                    DiscreteSpace(5),
                    BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                ]
            ),
            [RLTypes.DISCRETE, RLTypes.CONTINUOUS, RLTypes.IMAGE],
        ],
        [
            5,
            MultiSpace(
                [
                    BoxSpace((5, 1), 0, 4, np.uint64, EnvTypes.DISCRETE),
                    BoxSpace((5, 5), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                    BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                    BoxSpace((5, 4, 4, 3), 0, 255, np.uint8, type=EnvTypes.COLOR),
                ]
            ),
            [RLTypes.DISCRETE, RLTypes.CONTINUOUS, RLTypes.IMAGE, RLTypes.IMAGE],
            MultiSpace(
                [
                    DiscreteSpace(5),
                    BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                    BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                ]
            ),
            [RLTypes.DISCRETE, RLTypes.CONTINUOUS, RLTypes.IMAGE],
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
    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=MultiSpace(
                    [
                        DiscreteSpace(5),
                        BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                        BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                    ]
                ),
                observation_space=MultiSpace(
                    [
                        DiscreteSpace(5),
                        BoxSpace((5,), 0, 1, np.float32, EnvTypes.CONTINUOUS),
                        BoxSpace((4, 4, 3), 0, 255, np.uint8, type=EnvTypes.GRAY_3ch),
                    ]
                ),
            ),
        )
    )
    rl_config = TestConfig(
        set_base_action_type=RLBaseTypes.ANY,
        set_base_observation_type=RLBaseTypes.ANY,
    )
    rl_config.observation_mode = ObservationModes.ENV | ObservationModes.RENDER_IMAGE
    rl_config.window_length = window_length
    rl_config.setup(env)

    assert true_rl_obs_space == rl_config.observation_space
    assert len(true_rl_obs_space.spaces) == len(rl_config.observation_spaces_one_step)
    for i in range(len(rl_config.observation_spaces_one_step)):
        assert true_rl_obs_space.spaces[i] == rl_config.observation_spaces_one_step[i]
        assert true_rl_obs_types[i] == rl_config.observation_spaces_one_step[i].rl_type

    assert isinstance(rl_config.action_space, MultiSpace)
    assert true_rl_act_space == rl_config.action_space
    for i in range(len(true_rl_act_space.spaces)):
        assert true_rl_act_space.spaces[i] == rl_config.action_space.spaces[i]
        assert true_rl_act_types[i] == rl_config.action_space.spaces[i].rl_type


@pytest.mark.parametrize(
    """
    window_length,
    true_rl_obs_space,
    true_rl_obs_types,
    true_rl_act_space,
    true_rl_act_types,
    """,
    [
        [1, DiscreteSpace(5), RLTypes.DISCRETE, MultiSpace([DiscreteSpace(5)]), [RLTypes.DISCRETE]],
        [
            5,
            BoxSpace((5, 1), 0, 4, np.uint64, EnvTypes.DISCRETE),
            RLTypes.DISCRETE,
            MultiSpace([DiscreteSpace(5)]),
            [RLTypes.DISCRETE],
        ],
    ],
)
def test_setup_multi_space_single(
    window_length,
    true_rl_obs_space,
    true_rl_obs_types,
    true_rl_act_space,
    true_rl_act_types,
):
    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=MultiSpace([DiscreteSpace(5)]),
                observation_space=MultiSpace([DiscreteSpace(5)]),
            ),
        )
    )
    rl_config = TestConfig(
        set_base_action_type=RLBaseTypes.ANY,
        set_base_observation_type=RLBaseTypes.ANY,
    )
    rl_config.window_length = window_length
    rl_config.setup(env)

    assert true_rl_obs_space == rl_config.observation_space
    assert len(rl_config.observation_spaces_one_step) == 1
    assert true_rl_obs_space == rl_config.observation_spaces_one_step[0]

    assert isinstance(rl_config.action_space, MultiSpace)
    assert true_rl_act_space == rl_config.action_space
    for i in range(len(true_rl_act_space.spaces)):
        assert true_rl_act_space.spaces[i] == rl_config.action_space.spaces[i]
        assert true_rl_act_types[i] == rl_config.action_space.spaces[i].rl_type


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
