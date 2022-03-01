import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import gym
import gym.spaces
from srl.base.define import EnvObservationType


class EnvBase(gym.Env, metaclass=ABCMeta):
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # gym.Env
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> EnvObservationType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    # gym.Env
    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError()

    # gym.Env
    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        # return state, reward, done, info
        raise NotImplementedError()

    @abstractmethod
    def fetch_valid_actions(self) -> Any:
        raise NotImplementedError()

    # gym.Env
    @abstractmethod
    def render(self, mode: str = "human") -> Any:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()


class GymEnvWrapper(EnvBase):
    def __init__(self, env: gym.Env):
        self.env = env

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.UNKOWN

    @property
    def max_episode_steps(self) -> int:
        if hasattr(self.env, "_max_episode_steps"):
            return self.env._max_episode_steps  # type: ignore
        else:
            return 999_999

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Any:
        return self.env.reset()

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        return self.env.step(action)

    def fetch_valid_actions(self) -> Optional[list]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return [a for a in range(self.action_space.n)]  # type: ignore
        return None

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, state: Any) -> None:
        self.env = pickle.loads(state)


if __name__ == "__main__":
    pass
