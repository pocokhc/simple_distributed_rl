from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Tuple, Union

from srl.base.define import KeyBindType
from srl.base.info import Info
from srl.base.render import IRender
from srl.base.spaces.space import TActSpace, TActType, TObsSpace, TObsType

if TYPE_CHECKING:
    from srl.base.rl.worker import RLWorker


class EnvBase(IRender, Generic[TActSpace, TActType, TObsSpace, TObsType], ABC):
    # Set these in subclasses
    next_player: int = 0
    done_reason: str = ""
    info: Info = Info()

    @property
    @abstractmethod
    def action_space(self) -> TActSpace:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self) -> TObsSpace:
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
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> TObsType:
        """reset

        Args:
            seed: set RNG seed

        Return:
            state(TObsType): initial state
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: TActType) -> Tuple[TObsType, Union[float, List[float]], bool, bool]:
        """Take one step forward

        Args:
            action (TActType): next player action

        Returns:
            state (TObsType): State after step
            rewards (float | list[float]): Reward for each agent after 1 step.
            terminated (bool): An end flag within the MDP, indicating a general end.
            truncated (bool): Flags for termination outside of the MDP, such as timeout or exception termination.
        """
        raise NotImplementedError()

    def backup(self) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def get_invalid_actions(self, player_index: int = -1) -> List[TActType]:
        return []

    def action_to_str(self, action: Union[str, TActType]) -> str:
        return str(action)

    def get_key_bind(self) -> Optional[KeyBindType]:
        return None

    def make_worker(self, name: str, **kwargs) -> Optional["RLWorker"]:
        return None

    @property
    def unwrapped(self) -> Any:
        return self

    @property
    def render_interval(self) -> float:
        return 1000 / 60

    # --------------------------------
    # direct
    # --------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, TObsType]:
        """direct step
        外部で環境を動かしてpolicyだけ実行したい場合に実装します。
        これは学習で使う場合を想定していません。

        Returns:
            is_start_episode (bool): Whether this step is the first of an episode. Used to initialize variables
            state (TObsType): state
        """
        raise NotImplementedError()

    def decode_action(self, action: TActType) -> Any:
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

    def get_valid_actions(self, player_index: int = -1) -> List[TActType]:
        return self.action_space.get_valid_actions(self.get_invalid_actions(player_index))
