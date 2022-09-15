import copy
import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    name: str
    kwargs: Dict = field(default_factory=dict)

    # episode option
    max_episode_steps: int = -1
    episode_timeout: int = -1  # s
    skip_frames: int = 0

    # gym
    gym_prediction_by_simulation: bool = True

    # render option
    font_name: str = ""
    font_size: int = 12

    def make_env(self) -> "srl.base.env.base.EnvRun":
        from srl.base.env.registration import make

        return make(self)

    def _update_env_info(self, env: "EnvBase"):
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
