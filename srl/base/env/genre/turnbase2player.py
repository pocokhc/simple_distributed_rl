from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from srl.base.define import EnvAction, Info
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
    #  close
    #  render_terminal
    #  render_gui
    #  render_rgb_array
    #  get_invalid_actions
    #  action_to_str
    #  make_worker

    @property
    @abstractmethod
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def call_reset(self) -> np.ndarray:
        # state
        raise NotImplementedError()

    @abstractmethod
    def call_step(self, action: EnvAction) -> Tuple[np.ndarray, float, float, bool, Info]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 2

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.call_reset(), [self.player_index]

    def step(
        self,
        actions: List[EnvAction],
        player_indices: List[int],
    ) -> Tuple[np.ndarray, List[float], bool, List[int], Info]:
        action = actions[player_indices[0]]
        n_s, reward1, reward2, done, info = self.call_step(action)
        return n_s, [reward1, reward2], done, [self.player_index], info
