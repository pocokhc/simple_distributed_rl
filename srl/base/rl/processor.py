import re
from abc import ABC
from typing import List, Optional, Tuple

from srl.base.define import EnvAction, EnvObservation, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase


class Processor(ABC):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        original_env: object,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        return env_observation_space, env_observation_type

    def process_observation(
        self,
        observation: EnvObservation,
        original_env: object,
    ) -> EnvObservation:
        return observation

    def process_rewards(
        self,
        rewards: List[float],
        original_env: object,
    ) -> List[float]:
        return rewards


class RuleBaseProcessor(ABC):
    def process_policy_before(
        self,
        env: EnvRun,
        player_index: int,
    ) -> Optional[EnvAction]:  # Noneの場合は rl で予測する
        return None

    def process_policy_after(
        self,
        action: EnvAction,
        env: EnvRun,
        player_index: int,
    ) -> EnvAction:
        return action

    def policy_render(self, env: EnvRun, player_index: int) -> None:
        pass
