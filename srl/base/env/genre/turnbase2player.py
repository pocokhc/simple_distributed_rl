from abc import abstractmethod
from typing import List, Tuple

from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env import EnvBase


class TurnBase2Player(EnvBase):
    @abstractmethod
    def call_reset(self) -> Tuple[EnvObservationType, dict]:
        # state, info
        raise NotImplementedError()

    @abstractmethod
    def call_step(self, action: EnvActionType) -> Tuple[EnvObservationType, float, float, bool, dict]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 2

    def reset(self) -> Tuple[EnvObservationType, dict]:
        return self.call_reset()

    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], bool, dict]:
        n_s, reward1, reward2, done, info = self.call_step(action)

        return n_s, [reward1, reward2], done, info
