from typing import List, Tuple

from srl.base.define import EnvAction, EnvInvalidAction, EnvObservation, Info
from srl.base.env.base import EnvRun


class SinglePlayEnvWrapper(EnvRun):
    def __init__(self, env: EnvRun):
        assert env.player_num == 1
        super().__init__(env.env)

    # ------------------------------------
    # episode functions
    # ------------------------------------
    def reset(self) -> EnvObservation:
        super().reset()
        return self.state

    def step(self, action: EnvAction) -> Tuple[EnvObservation, float, bool, Info]:
        super().step([action])
        return self.state, self.step_rewards[0], self.done, self.info

    def get_invalid_actions(self, player_index: int = 0) -> List[EnvInvalidAction]:
        return super().get_invalid_actions(player_index)

    # ------------------------------------
    # util functions
    # ------------------------------------
    def sample(self) -> EnvAction:
        return super().sample(0)
