import logging
import pickle
from abc import ABC
from typing import TYPE_CHECKING, Any

from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun

logger = logging.getLogger(__name__)


class EnvProcessor(ABC):

    def remap_action_space(self, action_space: SpaceBase, env: "EnvRun") -> SpaceBase:
        return action_space

    def remap_observation_space(self, observation_space: SpaceBase, env: "EnvRun") -> SpaceBase:
        return observation_space

    def setup(self, env: "EnvRun"):
        pass

    # --- 実装されている場合に実行
    # def remap_reset(self, state, env: "EnvRun") -> None:
    #    return state

    # def remap_step_action(self, action, env: "EnvRun"):
    #    return action

    # def remap_step(self, state, rewards: List[float], terminated: bool, truncated: bool, env: "EnvRun"):
    #    return state, rewards, terminated, truncated

    # def remap_step_invalid_actions(self, invalid_actions, env: "EnvRun"):
    #    return invalid_actions

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
