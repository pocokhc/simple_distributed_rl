from dataclasses import dataclass

import pytest

import srl
from srl.base.define import EnvObservationTypes, RLBaseTypes, RLTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.rl.config import RLConfig
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase


class StubEnv(EnvBase):
    def __init__(
        self,
        action_space,
        observation_space,
        observation_type,
    ) -> None:
        self._action_space = action_space
        self._observation_space = observation_space
        self._observation_type = observation_type
        self.state = 0

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationTypes:
        return self._observation_type

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

    def getName(self) -> str:
        return "test"

    @property
    def base_action_type(self) -> RLBaseTypes:
        return self.set_base_action_type

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return self.set_base_observation_type

    def get_use_framework(self) -> str:
        return ""


_EUNK = EnvObservationTypes.UNKNOWN
_EDIS = EnvObservationTypes.DISCRETE
_ECON = EnvObservationTypes.CONTINUOUS
_ECOL = EnvObservationTypes.COLOR
_DS = DiscreteSpace(5)
_BS = BoxSpace((3,), -1, 1)


@pytest.mark.parametrize(
    """
    env_obs_type,
    env_obs_space,
    override_obs_type,
    use_image,
    window_length,
    rl_obs_type,
    rl_env_obs_type,
    rl_obs_space,
    """,
    [
        [_EDIS, _DS, _EUNK, False, 1, RLTypes.DISCRETE, _EDIS, _DS],
        [_EDIS, _DS, _EUNK, False, 4, RLTypes.CONTINUOUS, _EDIS, BoxSpace((4, 1), 0, 4)],
        [_EDIS, _DS, _ECON, False, 1, RLTypes.DISCRETE, _ECON, _DS],  # type mismatch
        [_EDIS, _DS, _ECON, False, 4, RLTypes.CONTINUOUS, _ECON, BoxSpace((4, 1), 0, 4)],
        [_EDIS, _DS, _EUNK, True, 1, RLTypes.IMAGE, _ECOL, BoxSpace((4, 4, 3), 0, 255)],
        [_EDIS, _DS, _EUNK, True, 4, RLTypes.IMAGE, _ECOL, BoxSpace((4, 4, 4, 3), 0, 255)],
        [_ECON, _BS, _EUNK, False, 1, RLTypes.CONTINUOUS, _ECON, BoxSpace((3,), -1, 1)],
        [_ECON, _BS, _EUNK, False, 4, RLTypes.CONTINUOUS, _ECON, BoxSpace((4, 3), -1, 1)],
    ],
)
@pytest.mark.parametrize(
    "env_act_space,rl_base_act_type,override_act_type,rl_act_type",
    [
        [_DS, RLBaseTypes.DISCRETE, RLTypes.UNKNOWN, RLTypes.DISCRETE],
        [_DS, RLBaseTypes.CONTINUOUS, RLTypes.UNKNOWN, RLTypes.CONTINUOUS],
        [_DS, RLBaseTypes.ANY, RLTypes.DISCRETE, RLTypes.DISCRETE],
        [_DS, RLBaseTypes.ANY, RLTypes.CONTINUOUS, RLTypes.CONTINUOUS],
        [_DS, RLBaseTypes.ANY, RLTypes.UNKNOWN, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.DISCRETE, RLTypes.UNKNOWN, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.CONTINUOUS, RLTypes.UNKNOWN, RLTypes.CONTINUOUS],
        [_BS, RLBaseTypes.ANY, RLTypes.DISCRETE, RLTypes.DISCRETE],
        [_BS, RLBaseTypes.ANY, RLTypes.CONTINUOUS, RLTypes.CONTINUOUS],
        [_BS, RLBaseTypes.ANY, RLTypes.UNKNOWN, RLTypes.CONTINUOUS],
    ],
)
def test_reset(
    # obs
    env_obs_type,
    env_obs_space,
    override_obs_type,
    use_image,
    window_length,
    rl_obs_type,
    rl_env_obs_type,
    rl_obs_space,
    # act
    env_act_space,
    rl_base_act_type,
    override_act_type,
    rl_act_type,
):
    env = srl.make_env(
        srl.EnvConfig(
            "config_StubEnv",
            dict(
                action_space=env_act_space,
                observation_space=env_obs_space,
                observation_type=env_obs_type,
            ),
        )
    )
    # base_observation_typeはANYでspaceの状態が反映される場合のみを確認
    rl_config = TestConfig(set_base_action_type=rl_base_act_type, set_base_observation_type=RLBaseTypes.ANY)
    rl_config.override_env_observation_type = override_obs_type
    rl_config.override_action_type = override_act_type
    rl_config.use_render_image_for_observation = use_image
    rl_config.window_length = window_length
    rl_config.setup(env)

    assert rl_act_type == rl_config.action_type
    assert rl_obs_type == rl_config.observation_type
    assert rl_env_obs_type == rl_config.env_observation_type
    assert rl_obs_space == rl_config.observation_space


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
