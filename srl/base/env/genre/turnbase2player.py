from typing import Generic, List, Optional, Tuple

from srl.base.define import TActType, TObsType
from srl.base.env import EnvBase


class TurnBase2Player(Generic[TActType, TObsType], EnvBase[TActType, TObsType]):
    def call_reset(self) -> Tuple[TObsType, dict]:
        # state, info
        raise NotImplementedError()

    def call_step(self, action: TActType) -> Tuple[TObsType, float, float, bool, dict]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 2

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> TObsType:
        state, info = self.call_reset()
        self.info.set_dict(info)
        return state

    def step(self, action: TActType) -> Tuple[TObsType, List[float], bool, bool]:
        n_s, reward1, reward2, done, info = self.call_step(action)
        self.info.set_dict(info)
        return n_s, [reward1, reward2], done, False
