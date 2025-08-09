import logging
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Optional

from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


@dataclass
class RLProcessor(ABC):
    """継承先も必ずdataclassで実装してください
    reward   : float->float の変換で以上のことが今のところなく、worker側の処理で十分なので実装なしに
    done     : env側の影響もあるため実装が困難
    statefull: 今のところ必要ないので実装なしに(ある場合はsetup/on_reset/backup/restoreが必要)
    """

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        """新しいSpaceを返す。適用しない場合はNoneを返す"""
        return None

    # --- 実装されている場合に実行
    # def remap_observation(self, state: EnvObservationType, prev_space: SpaceBase, new_space: SpaceBase, **kwargs) -> EnvObservationType:
    #    return state

    def copy(self) -> "RLProcessor":
        o = self.__class__()

        for k, v in self.__dict__.items():
            try:
                setattr(o, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        return o
