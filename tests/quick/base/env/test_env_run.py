import time
from pprint import pprint
from typing import cast

import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import SpaceTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.env.processor import EnvProcessor
from srl.base.exception import SRLError
from srl.base.spaces.discrete import DiscreteSpace


class StubProcessor(EnvProcessor):
    def __init__(self) -> None:
        self.n = 0

    def remap_step(self, state, rewards, done, info, env: EnvRun):
        self.n += 1
        return state, rewards, done, info

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

    def reset(self):
        self._step = 0
        state = 0
        info = {}
        return state, info

    def step(self, action):
        self._step += 1
        next_state = 1
        rewards = [1]
        done = False
        info = {"action": action}
        return next_state, rewards, done, info

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

    env.setup(RunContext())
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

    env.setup(RunContext())
    env.reset()
    time.sleep(2)
    env.step(0)
    assert env.done

    with pytest.raises(SRLError):
        env.step(0)


def test_backup():
    env_config = srl.EnvConfig("base_StubEnv", processors=[StubProcessor()])
    env = env_config.make()
    env.setup()
    env.reset()

    env.step(0)
    assert env.episode_rewards[0] == 1
    assert env.get_invalid_actions() == [1]
    assert env.get_valid_actions() == [0, 2, 3]
    assert env.unwrapped._step == 1
    assert cast(StubProcessor, env._processors[0]).n == 1
    backup1 = env.backup()

    env.step(1)
    assert env.episode_rewards[0] == 2
    assert env.get_invalid_actions() == [2]
    assert env.get_valid_actions() == [0, 1, 3]
    assert cast(StubProcessor, env._processors[0]).n == 2
    assert env.unwrapped._step == 2

    pprint(backup1)
    env.restore(backup1)
    assert env.episode_rewards[0] == 1
    assert env.get_invalid_actions() == [1]
    assert env.get_valid_actions() == [0, 2, 3]
    assert env.unwrapped._step == 1
    assert cast(StubProcessor, env._processors[0]).n == 1
