from abc import abstractmethod
from typing import List, Tuple

import numpy as np
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
    #  close
    #  render_terminal
    #  render_gui
    #  render_rgb_array
    #  get_invalid_actions
    #  action_to_str
    #  make_worker

    @abstractmethod  # new method
    def call_reset(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod  # new method
    def call_step(self, action: EnvAction) -> Tuple[np.ndarray, float, bool, Info]:
        raise NotImplementedError()

    # -----------------------------------------------------
    #  inherit implementation(継承元の実装)
    # -----------------------------------------------------
    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.call_reset(), [0]

    def step(
        self,
        actions: List[EnvAction],
        player_indices: List[int],
    ) -> Tuple[EnvObservation, List[float], bool, List[int], Info]:
        n_state, reward, done, info = self.call_step(actions[0])
        return n_state, [reward], done, [0], info
