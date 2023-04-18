import warnings
from abc import abstractmethod
from typing import List, Tuple

from srl.base.define import EnvAction, EnvObservation, Info
from srl.base.env import EnvBase


class SinglePlayEnv(EnvBase):
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
    #  action_to_str
    #  make_worker
    #  set_seed

    @abstractmethod
    def call_reset(self) -> Tuple[EnvObservation, Info]:
        # state, info
        raise NotImplementedError()

    @abstractmethod
    def call_step(self, action: EnvAction) -> Tuple[EnvObservation, float, bool, Info]:
        # state, reward, done, info
        raise NotImplementedError()

    def call_get_invalid_actions(self) -> List[int]:
        return []

    def call_direct_reset(self, *args, **kwargs) -> Tuple[EnvObservation, Info]:
        # state, info
        raise NotImplementedError()

    def call_direct_step(self, *args, **kwargs) -> Tuple[EnvObservation, float, bool, Info]:
        # state, reward, done, info
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[EnvObservation, int, Info]:
        state = self.call_reset()
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state, info = state
        else:
            info = {}
            warnings.warn("The return value of reset has changed from 'state' to 'state, info'.", DeprecationWarning)
        return state, 0, info

    def step(
        self,
        action: EnvAction,
        player_index: int,
    ) -> Tuple[EnvObservation, List[float], bool, int, Info]:
        n_state, reward, done, info = self.call_step(action)
        return n_state, [reward], done, 0, info

    def get_invalid_actions(self, player_index: int) -> List[int]:
        return self.call_get_invalid_actions()

    def direct_reset(self, *args, **kwargs) -> Tuple[EnvObservation, int, Info]:
        state = self.call_direct_reset(*args, **kwargs)
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state, info = state
        else:
            info = {}
            warnings.warn("The return value of reset has changed from 'state' to 'state, info'.", DeprecationWarning)
        return state, 0, info

    def direct_step(self, *args, **kwargs) -> Tuple[EnvObservation, List[float], bool, int, Info]:
        n_state, reward, done, info = self.call_direct_step(*args, **kwargs)
        return n_state, [reward], done, 0, info
