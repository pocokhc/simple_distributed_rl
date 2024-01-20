from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from srl.base.define import DoneTypes, EnvObservationTypes
from srl.base.env.base import SpaceBase

if TYPE_CHECKING:
    import gym
    import gymnasium


class GymUserWrapper(ABC):
    def action_space(
        self,
        action_space: Optional[SpaceBase],
        env: Union["gymnasium.Env", "gym.Env"],
    ) -> Optional[SpaceBase]:
        return action_space

    def action(self, action: Any, env: Union["gymnasium.Env", "gym.Env"]) -> Any:
        return action

    def observation_space(
        self,
        observation_type: EnvObservationTypes,
        observation_space: Optional[SpaceBase],
        env: Union["gymnasium.Env", "gym.Env"],
    ) -> Tuple[EnvObservationTypes, Optional[SpaceBase]]:
        return observation_type, observation_space

    def observation(self, observation: Any, env: Union["gymnasium.Env", "gym.Env"]) -> Any:
        return observation

    def reward(self, reward: float, env: Union["gymnasium.Env", "gym.Env"]) -> float:
        return reward

    def done(self, done: DoneTypes, env: Union["gymnasium.Env", "gym.Env"]) -> Union[bool, DoneTypes]:
        return done
