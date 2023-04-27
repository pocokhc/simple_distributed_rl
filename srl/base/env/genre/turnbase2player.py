import warnings
from abc import abstractmethod
from typing import List, Tuple

from srl.base.define import EnvAction, EnvObservation, Info
from srl.base.env import EnvBase


class TurnBase2Player(EnvBase):
    # -----------------------------------------------------
    # inheritance target implementation(継承先に必要な実装)
    # -----------------------------------------------------
    # @property (define parent)
    #  action_space : SpaceBase
    #  observation_space : SpaceBase
    #  observation_type  : EnvObservationType
    #  max_episode_steps : int
    #
    # functions (define parent)
    # (abstractmethod)
    #  backup
    #  restore
    # (option)
    #  render_terminal
    #  render_rgb_array
    #  close
    #  get_invalid_actions
    #  action_to_str
    #  make_worker
    #  set_seed

    @abstractmethod
    def call_reset(self) -> Tuple[EnvObservation, Info]:
        # state, info
        raise NotImplementedError()

    @abstractmethod
    def call_step(self, action: EnvAction) -> Tuple[EnvObservation, float, float, bool, Info]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 2

    def reset(self) -> Tuple[EnvObservation, Info]:
        state = self.call_reset()
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state, info = state
        else:
            info = {}
            warnings.warn("The return value of reset has changed from (state) to (state, info).", DeprecationWarning)
        return state, info

    def step(self, action: EnvAction) -> Tuple[EnvObservation, List[float], bool, Info]:
        n_s, reward1, reward2, done, info = self.call_step(action)

        return n_s, [reward1, reward2], done, info
