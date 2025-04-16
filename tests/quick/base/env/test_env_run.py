import time
from pprint import pprint
from typing import Optional, cast

import pytest

import srl
from srl.base.define import SpaceTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.env.processor import EnvProcessor
from srl.base.exception import SRLError
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase


class StubProcessor(EnvProcessor):
    def __init__(self) -> None:
        self.n = 0
        self._remap_action_space = 0
        self._remap_observation_space = 0
        self._setup = 0
        self._remap_reset = 0
        self._remap_action = 0
        self._remap_invalid_actions = 0
        self._remap_observation = 0

    def remap_action_space(self, prev_space: SpaceBase, env_run: EnvRun) -> Optional[SpaceBase]:
        self._remap_action_space += 1
        return prev_space

    def remap_observation_space(self, prev_space: SpaceBase, env_run: EnvRun) -> Optional[SpaceBase]:
        self._remap_observation_space += 1
        return prev_space

    def setup(self, env_run: EnvRun):
        self._setup += 1

    def remap_reset(self, env_run: EnvRun):
        self._remap_reset += 1

    def remap_action(self, action, prev_space: SpaceBase, new_space: SpaceBase, env_run: EnvRun):
        self._remap_action += 1
        return action

    def remap_invalid_actions(self, invalid_actions, prev_space: SpaceBase, new_space: SpaceBase, env_run: EnvRun):
        self._remap_invalid_actions += 1
        return invalid_actions

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, env_run: EnvRun):
        self._remap_observation += 1
        return state

    def remap_step(self, rewards, done, info, env_run: EnvRun):
        self.n += 1
        return rewards, done, info

    def backup(self):
        return self.n

    def restore(self, d):
        self.n = d


class StubEnv(EnvBase):
    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(4)

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(4)

    @property
    def observation_type(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def player_num(self) -> int:
        return 1

    @property
    def next_player_index(self) -> int:
        return 0

    def setup(self, render_mode, **kwargs):
        self.render_mode = render_mode

    def reset(self, **kwargs):
        self._step = 0
        state = 0
        return state

    def step(self, action):
        self._step += 1
        next_state = 1
        rewards = [1]
        terminated = False
        truncated = False
        self.info["action"] = action
        return next_state, rewards, terminated, truncated

    # backup/restore で現環境を復元できるように実装
    def backup(self):
        return self._step

    def restore(self, data) -> None:
        self._step = data

    def close(self):
        raise ValueError("TestError")

    def get_invalid_actions(self, player_index: int):
        return [self._step % 4]

    def render_terminal(self):
        print(self._step)


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    registration.register(id="base_StubEnv", entry_point=__name__ + ":StubEnv", check_duplicate=False)
    yield


def test_base():
    env_config = srl.EnvConfig("base_StubEnv", frameskip=3)
    env = env_config.make()
    env_org = cast(StubEnv, env.unwrapped)

    with pytest.raises(SRLError):
        env.reset()

    with pytest.raises(SRLError):
        env.step(0)

    env.setup()
    assert env_org.render_mode == ""
    env.reset()
    assert env.step_num == 0

    env.step(0)
    assert env.step_num == 1
    assert env_org._step == 4
    assert env.get_invalid_actions() == [0]
    assert env.get_valid_actions() == [1, 2, 3]
    env2 = srl.make_env("base_StubEnv")
    env2.restore(env.backup())
    assert env2.step_num == 1
    assert cast(StubEnv, env2.unwrapped)._step == 4


def test_max_steps():
    env_config = srl.EnvConfig("base_StubEnv", max_episode_steps=10)
    env = env_config.make()

    env.setup()
    env.reset()
    for _ in range(10):
        env.step(0)

    assert not env.done
    env.step(0)
    assert env.done

    with pytest.raises(SRLError):
        env.step(0)


def test_timeout():
    env_config = srl.EnvConfig("base_StubEnv", episode_timeout=1)
    env = env_config.make()

    env.setup()
    env.reset()
    time.sleep(2)
    env.step(0)
    assert env.done

    with pytest.raises(SRLError):
        env.step(0)


def test_processor_backup():
    env_config = srl.EnvConfig("base_StubEnv", processors=[StubProcessor()])
    env = env_config.make()
    proc = cast(StubProcessor, env._processors[0])

    env.setup()
    env.reset()

    env.step(0)
    assert env.episode_rewards[0] == 1
    assert env.get_invalid_actions() == [1]
    assert env.get_valid_actions() == [0, 2, 3]
    assert env.unwrapped._step == 1
    assert proc.n == 1
    backup1 = env.backup()

    env.step(1)
    assert env.episode_rewards[0] == 2
    assert env.get_invalid_actions() == [2]
    assert env.get_valid_actions() == [0, 1, 3]
    assert proc.n == 2
    assert env.unwrapped._step == 2

    # assert processor
    assert proc._remap_action_space == 1
    assert proc._remap_observation_space == 1
    assert proc._setup == 1
    assert proc._remap_reset == 1
    assert proc._remap_action == 2
    assert proc._remap_invalid_actions == 2
    assert proc._remap_observation == 3

    # assert restore
    pprint(backup1)
    env.restore(backup1)
    assert env.episode_rewards[0] == 1
    assert env.get_invalid_actions() == [1]
    assert env.get_valid_actions() == [0, 2, 3]
    assert env.unwrapped._step == 1
    assert cast(StubProcessor, env._processors[0]).n == 1
