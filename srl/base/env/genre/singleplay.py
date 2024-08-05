from typing import Generic, List, Optional, Tuple

from srl.base.define import RLActionType, TActType, TObsType
from srl.base.env import EnvBase


class SinglePlayEnv(Generic[TActType, TObsType], EnvBase[TActType, TObsType]):
    def call_reset(self) -> Tuple[TObsType, dict]:
        # state, info
        raise NotImplementedError()

    def call_step(self, action: TActType) -> Tuple[TObsType, float, bool, dict]:
        # state, reward, done, info
        raise NotImplementedError()

    def call_get_invalid_actions(self) -> List[RLActionType]:
        return []

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 1

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> TObsType:
        state, info = self.call_reset()
        self.info.set_dict(info)
        return state

    def step(self, action: TActType) -> Tuple[TObsType, List[float], bool, bool]:
        n_state, reward, done, info = self.call_step(action)
        self.info.set_dict(info)
        return n_state, [reward], done, False

    def get_invalid_actions(self, player_index: int) -> List[RLActionType]:
        return self.call_get_invalid_actions()
