from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from srl.base.define import DoneTypes, EnvActionType, EnvObservationType, KeyBindType
from srl.base.render import IRender
from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.rl.worker import RLWorker


class EnvBase(ABC, IRender):
    # --------------------------------
    # implement
    # --------------------------------
    @property
    @abstractmethod
    def action_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def player_num(self) -> int:
        raise NotImplementedError()

    @property
    def reward_info(self) -> dict:
        return {
            "min": None,
            "max": None,
            "baseline": None,
        }

    def setup(self, **kwargs) -> None:
        """run.core_play の文脈で最初に呼ばれます
        引数は RunContext の変数が入ります
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[EnvObservationType, dict]:
        """reset

        Returns: init_state, info
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], Union[bool, DoneTypes], dict]:
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

    def backup(self) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def get_invalid_actions(self, player_index: int = -1) -> List[EnvObservationType]:
        return []

    def action_to_str(self, action: Union[str, EnvActionType]) -> str:
        return str(action)

    def get_key_bind(self) -> Optional[KeyBindType]:
        return None

    def make_worker(self, name: str, **kwargs) -> Optional["RLWorker"]:
        return None

    @property
    def unwrapped(self) -> Any:
        return self

    def set_seed(self, seed: Optional[int] = None) -> None:
        pass

    @property
    def render_interval(self) -> float:
        return 1000 / 60

    # --------------------------------
    # direct
    # --------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, EnvObservationType, int, dict]:
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

    def get_valid_actions(self, player_index: int = -1) -> List[EnvObservationType]:
        return self.action_space.get_valid_actions(self.get_invalid_actions(player_index))
