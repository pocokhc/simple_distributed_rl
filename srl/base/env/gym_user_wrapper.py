from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    import gym
    import gymnasium


class GymUserWrapper(ABC):
    def remap_action_space(self, env: Union["gymnasium.Env", "gym.Env"]) -> Optional[SpaceBase]:
        return None

    def remap_observation_space(self, env: Union["gymnasium.Env", "gym.Env"]) -> Optional[SpaceBase]:
        return None

    def remap_action(self, action: Any, env: Union["gymnasium.Env", "gym.Env"]) -> Any:
        return action

    def remap_observation(self, observation: Any, env: Union["gymnasium.Env", "gym.Env"]) -> Any:
        return observation

    def remap_reward(self, reward: float, env: Union["gymnasium.Env", "gym.Env"]) -> float:
        return reward

    def remap_done(
        self, terminated: bool, truncated: bool, env: Union["gymnasium.Env", "gym.Env"]
    ) -> Tuple[bool, bool]:
        return terminated, truncated
