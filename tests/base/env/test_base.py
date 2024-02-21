import time
from typing import cast

import pytest

import srl
from srl.base.define import EnvObservationTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.spaces.discrete import DiscreteSpace


class StubEnv(EnvBase):
    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(4)

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(4)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

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
        info = {}
        return next_state, rewards, done, info

    # backup/restore で現環境を復元できるように実装
    def backup(self):
        return self._step

    def restore(self, data) -> None:
        self._step = data

    def close(self):
        raise ValueError("TestError")

    def get_invalid_actions(self, player_index: int):
        return [1]

    def render_terminal(self):
        print(self._step)


registration.register(id="base_StubEnv", entry_point=__name__ + ":StubEnv")


def test_EnvRun():
    env_config = srl.EnvConfig("base_StubEnv", frameskip=3)
    env = srl.make_env(env_config)
    env_org = cast(StubEnv, env.unwrapped)

    with pytest.raises(AssertionError):
        env.step(0)

    env.reset()
    assert env.step_num == 0

    env.step(0)
    assert env.step_num == 1
    assert env_org._step == 4
    assert env.get_invalid_actions() == [1]
    assert env.get_valid_actions() == [0, 2, 3]
    env2 = srl.make_env("base_StubEnv")
    env2.restore(env.backup())
    assert env2.step_num == 1
    assert cast(StubEnv, env2.unwrapped)._step == 4


def test_EnvRun_max_steps():
    env_config = srl.EnvConfig("base_StubEnv", max_episode_steps=10)
    env = srl.make_env(env_config)

    env.reset()
    for _ in range(10):
        env.step(0)

    assert not env.done
    env.step(0)
    assert env.done

    with pytest.raises(AssertionError):
        env.step(0)


def test_EnvRun_timeout():
    env_config = srl.EnvConfig("base_StubEnv", episode_timeout=1)
    env = srl.make_env(env_config)

    env.reset()
    time.sleep(2)
    env.step(0)
    assert env.done

    with pytest.raises(AssertionError):
        env.step(0)
