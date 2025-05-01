import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Tuple, Union

from srl.base.define import KeyBindType
from srl.base.info import Info
from srl.base.render import IRender
from srl.base.spaces.space import TActSpace, TActType, TObsSpace, TObsType

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.worker import RLWorker

logger = logging.getLogger(__name__)


class EnvBase(IRender, Generic[TActSpace, TActType, TObsSpace, TObsType], ABC):
    def __init__(self) -> None:
        self.init_base(env_run=None)

    def __post_init__(self) -> None:
        # dataclass用
        self.init_base(env_run=None)

    def init_base(self, env_run: Optional["EnvRun"]):
        """
        互換性のために別途初期化関数を定義
        registration内で手動で呼び出し
        """
        # Set these in subclasses
        if hasattr(self, "next_player"):
            if not isinstance(self.next_player, int):
                logger.warning(f"[env_base.init] 'next_player' type is not 'int'. {self.next_player}({type(self.next_player)})")
        else:
            self.next_player: int = 0

        if hasattr(self, "done_reason"):
            if not isinstance(self.done_reason, str):
                logger.warning(f"[env_base.init] 'done_reason' type is not 'str'. {self.done_reason}({type(self.done_reason)})")
        else:
            self.done_reason: str = ""

        if hasattr(self, "info"):
            if not isinstance(self.info, Info):
                logger.warning(f"[env_base.init] 'info' type is not 'Info'. {self.info}({type(self.info)})")
        else:
            self.info: Info = Info()

        self.env_run = env_run
        return self

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
    def reward_range(self) -> Tuple[float, float]:
        """rewardの取りうる範囲"""
        return (-math.inf, math.inf)

    @property
    def reward_baseline(self):
        """学習されたと見なされる報酬の閾値を返す、仕様が固まり切っていないので仮"""
        return None

    def setup(self, **kwargs) -> None:
        """run.core_play の文脈で最初に呼ばれます
        引数は RunContext の変数が入ります
        """
        pass

    def teardown(self, **kwargs) -> None:
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

    # ------------------------------------
    # context
    # ------------------------------------
    @property
    def training(self) -> bool:
        if self.env_run is not None:
            return self.env_run.context.training
        return False

    @property
    def rendering(self) -> bool:
        if self.env_run is not None:
            return self.env_run.context.env_render_mode != ""
        return False

    # --------------------------------
    # direct
    # --------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, TObsType, bool]:
        """direct step
        外部で環境を動かしてpolicyだけ実行したい場合に実装します。
        これは学習で使う場合を想定していません。

        Returns:
            is_start_episode (bool): Whether this step is the first of an episode. Used to initialize variables
            state (TObsType): state
            is_end_episode (bool): Flag indicating whether the episode is over
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
