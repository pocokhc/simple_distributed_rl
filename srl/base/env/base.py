from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from srl.base.define import (
    DoneTypes,
    EnvActionType,
    EnvObservationType,
    EnvObservationTypes,
    InfoType,
    InvalidActionsType,
    KeyBindType,
)
from srl.base.render import IRender
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.rl.base import RLWorker


class EnvBase(ABC, IRender):
    # --------------------------------
    # implement properties
    # --------------------------------

    # --- action
    @property
    @abstractmethod
    def action_space(self) -> SpaceBase:
        raise NotImplementedError()

    # --- observation
    @property
    @abstractmethod
    def observation_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> EnvObservationTypes:
        raise NotImplementedError()

    # --- properties
    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def player_num(self) -> int:
        raise NotImplementedError()

    # --- reward(option)
    @property
    def reward_info(self) -> dict:
        return {
            "min": None,
            "max": None,
            "baseline": None,
        }

    @property
    def info_types(self) -> dict:
        """infoの情報のタイプを指定、出力形式等で使用を想定
        各行の句は省略可能
        name : {
            "type": 型を指定(None, int, float, str)
            "data": 以下のデータ形式を指定
                "ave" : 平均値を使用(default)
                "last": 最後のデータを使用
                "min" : 最小値
                "max" : 最大値
        }
        """
        return {}  # NotImplemented

    # --------------------------------
    # implement functions
    # --------------------------------
    @abstractmethod
    def reset(self) -> Tuple[EnvObservationType, InfoType]:
        """reset

        Returns: init_state, info
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], Union[bool, DoneTypes], InfoType]:
        """step

        Args:
            action (EnvAction): player_index action

        Returns:(
            next_state,
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done or DoneTypes,
            info,
        )
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def next_player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    # --------------------------------
    # options
    # --------------------------------
    def close(self) -> None:
        pass

    def get_invalid_actions(self, player_index: int = -1) -> InvalidActionsType:
        return []

    def action_to_str(self, action: Union[str, EnvActionType]) -> str:
        return str(action)

    def get_key_bind(self) -> KeyBindType:
        return None

    def make_worker(self, name: str, **kwargs) -> Optional["RLWorker"]:
        return None

    def get_original_env(self) -> Any:
        return self

    def set_seed(self, seed: Optional[int] = None) -> None:
        pass

    @property
    def render_interval(self) -> float:
        return 1000 / 60

    # --------------------------------
    # direct
    # --------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, EnvObservationType, int, InfoType]:
        """direct step
        外部で環境を動かしてpolicyだけ実行したい場合に実装します。
        これは学習で使う場合を想定していません。

        Returns:(
            is_start_episode,
            state,
            player_index,
            info,
        )
        """
        raise NotImplementedError()

    def decode_action(self, action: EnvActionType) -> Any:
        raise NotImplementedError()

    @property
    def can_simulate_from_direct_step(self) -> bool:
        """
        direct_stepで実行した場合に、そのあとにstepでシミュレーション可能かどうかを返します。
        direct_step後にstepを機能させるには、direct_step内でstepが実行できるまでenv環境を復元する必要があります。
        主にMCTS等、シミュレーションが必要なアルゴリズムに影響があります。
        """
        raise NotImplementedError()

    # --------------------------------
    # utils
    # --------------------------------
    def copy(self):
        env = self.__class__()
        env.restore(self.backup())
        return env

    def get_valid_actions(self, player_index: int = -1) -> InvalidActionsType:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            return [a for a in range(self.action_space.n) if a not in invalid_actions]
        else:
            return []
