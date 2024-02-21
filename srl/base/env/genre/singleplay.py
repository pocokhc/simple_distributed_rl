from abc import abstractmethod
from typing import List, Tuple

from srl.base.define import EnvActionType, EnvObservationType, InfoType, RLInvalidActionType
from srl.base.env import EnvBase


class SinglePlayEnv(EnvBase):
    @abstractmethod
    def call_reset(self) -> Tuple[EnvObservationType, InfoType]:
        # state, info
        raise NotImplementedError()

    @abstractmethod
    def call_step(self, action: EnvActionType) -> Tuple[EnvObservationType, float, bool, InfoType]:
        # state, reward, done, info
        raise NotImplementedError()

    def call_get_invalid_actions(self) -> List[RLInvalidActionType]:
        return []

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 1

    @property
    def next_player_index(self) -> int:
        return 0

    def reset(self) -> Tuple[EnvObservationType, InfoType]:
        return self.call_reset()

    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], bool, InfoType]:
        n_state, reward, done, info = self.call_step(action)
        return n_state, [reward], done, info

    def get_invalid_actions(self, player_index: int) -> List[RLInvalidActionType]:
        return self.call_get_invalid_actions()
