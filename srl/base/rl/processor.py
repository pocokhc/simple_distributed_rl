import logging
import pickle
from abc import ABC
from typing import TYPE_CHECKING, Any

from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


class RLProcessor(ABC):
    def setup(self, env: "EnvRun", rl_config: "RLConfig"):
        pass

    def remap_observation_space(
        self, env_observation_space: SpaceBase, env: "EnvRun", rl_config: "RLConfig"
    ) -> SpaceBase:
        return env_observation_space

    # --- 実装されている場合に実行
    # def remap_on_reset(self, worker: "WorkerRun", env: "EnvRun") -> None:
    #    pass

    # def remap_observation(self, state: EnvObservationType, worker: "WorkerRun", env: "EnvRun") -> EnvObservationType:
    #    return state

    # def remap_reward(self, reward: float, worker: "WorkerRun", env: "EnvRun") -> float:
    #    return reward

    # doneはenv側も影響するため定義が難しい

    def backup(self) -> Any:
        return None

    def restore(self, dat: Any) -> None:
        pass
    
    def copy(self) -> "RLProcessor":
        o = self.__class__()

        for k, v in self.__dict__.items():
            try:
                setattr(o, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        return o
