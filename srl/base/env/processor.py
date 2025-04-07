import logging
import pickle
from abc import ABC
from typing import Any, Optional

from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class EnvProcessor(ABC):
    def remap_action_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        """新しいSpaceを返す。適用しない場合はNoneを返す"""
        return None

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        """新しいSpaceを返す。適用しない場合はNoneを返す"""
        return None

    def setup(self, **kwargs):
        pass

    # --- 実装されている場合に実行
    # def remap_reset(self, **kwargs) -> None:
    #    return state

    # def remap_action(self, action, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
    #    return action

    # def remap_invalid_actions(self, invalid_actions, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
    #    return invalid_actions

    # def remap_observation(self, state: EnvObservationType, prev_space: SpaceBase, new_space: SpaceBase, **kwargs) -> EnvObservationType:
    #    return state

    # def remap_step(self, rewards: List[float], terminated: bool, truncated: bool, **kwargs):
    #    return rewards, terminated, truncated

    def backup(self) -> Any:
        return None

    def restore(self, dat: Any) -> None:
        pass

    def copy(self) -> "EnvProcessor":
        o = self.__class__()

        for k, v in self.__dict__.items():
            try:
                setattr(o, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        return o
