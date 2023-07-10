import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import gym
    import gymnasium

    from srl.base.env.base import EnvBase
    from srl.base.env.env_run import EnvRun
    from srl.base.env.gym_user_wrapper import GymUserWrapper


logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    name: str
    kwargs: Dict = field(default_factory=dict)

    # --- episode option
    max_episode_steps: int = -1
    episode_timeout: int = -1  # s
    frameskip: int = 0

    # --- gym
    gymnasium_make_func: Optional[Callable[..., "gymnasium.Env"]] = None
    gym_make_func: Optional[Callable[..., "gym.Env"]] = None
    gym_check_image: bool = True
    gym_prediction_by_simulation: bool = True
    gym_prediction_step: int = 10
    gym_wrappers: List["GymUserWrapper"] = field(default_factory=list)

    # --- render option
    render_interval: float = -1
    """ ms. -1 is the interval set by Env.  """
    render_scale: float = 1.0
    font_name: str = ""
    font_size: int = 12

    # --- other
    enable_sanitize_value: bool = True
    enable_assertion_value: bool = False

    def make_env(self) -> "EnvRun":
        from srl.base.env.registration import make

        return make(self)

    def _update_env_info(self, env: "EnvBase"):
        """env 作成時に env 情報を元に Config を更新"""
        if self.max_episode_steps <= 0:
            self.max_episode_steps = env.max_episode_steps
        self.player_num = env.player_num

    def copy(self) -> "EnvConfig":
        config = EnvConfig(self.name)
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
            elif type(v) in [list, dict]:
                setattr(config, k, copy.deepcopy(v))
        return config
