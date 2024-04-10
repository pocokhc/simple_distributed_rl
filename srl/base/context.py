import copy
import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from srl.base.define import PlayerType, RenderModes
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


class RunNameTypes(enum.Enum):
    main = enum.auto()
    trainer = enum.auto()
    actor = enum.auto()
    eval = enum.auto()


@dataclass
class RunContext:
    """
    実行時の状態をまとめたクラス
    A class that summarizes the runtime state
    """

    players: List[PlayerType] = field(default_factory=list)

    # --- play context
    run_name: RunNameTypes = RunNameTypes.main
    # stop config
    max_episodes: int = 0
    timeout: float = 0
    max_steps: int = 0
    max_train_count: int = 0
    max_memory: int = 0
    # play config
    shuffle_player: bool = True
    disable_trainer: bool = False
    # play info
    distributed: bool = False
    training: bool = False
    rendering: bool = False
    render_mode: Union[str, RenderModes] = RenderModes.none

    # --- mp
    actor_id: int = 0
    actor_num: int = 1

    # --- random
    seed: Optional[int] = None

    # --- device
    framework: str = ""
    enable_tf_device: bool = True
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    def to_dict(self) -> dict:
        return convert_for_json(self.__dict__)

    def copy(self) -> "RunContext":
        return copy.deepcopy(self)

    def set_device(
        self,
        framework: str,
        used_device_tf: str,
        used_device_torch: str,
        rl_config: "RLConfig",
    ):
        self.framework = framework
        self.used_device_tf = used_device_tf
        self.used_device_torch = used_device_torch
        rl_config._used_device_tf = used_device_tf
        rl_config._used_device_torch = used_device_torch
