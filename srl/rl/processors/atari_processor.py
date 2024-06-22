import numpy as np

from srl.base.define import DoneTypes, SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.env.processor import Processor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.functions import image_processor
from srl.utils.common import is_package_installed


class AtariProcessor(Processor):
    def __init__(self):
        assert is_package_installed("ale_py")

    def remap_reset(self, env: EnvRun):
        self.lives = env.unwrapped.env.unwrapped.ale.lives()

    def remap_step(self, state, rewards, done, info, env: EnvRun):
        new_lives = env.unwrapped.env.unwrapped.ale.lives()
        if new_lives < self.lives:
            return state, rewards, DoneTypes.TERMINATED, info
        self.lives = new_lives
        return state, rewards, done, info


class AtariPongProcessor(Processor):
    def __init__(self):
        assert is_package_installed("ale_py")

    def remap_observation_space(self, observation_space: SpaceBase, env: EnvRun) -> SpaceBase:
        return BoxSpace((84, 84), 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch)

    def remap_reset(self, state, info, env: EnvRun):
        self.point = 0
        return self._remap_state(state), info

    def remap_step(self, state, rewards, done, info, env: EnvRun):
        if env.reward == 1:
            self.point += 1
        elif env.reward == -1:
            self.point += 1
        if self.point == 5:
            done = DoneTypes.TERMINATED
        return self._remap_state(state), rewards, done, info

    def _remap_state(self, state):
        state = image_processor(
            state,
            SpaceTypes.COLOR,
            SpaceTypes.GRAY_2ch,
            trimming=(35, 10, 195, 150),
            resize=(84, 84),
        )
        state = np.where(state > 127, 255, 0).astype(np.uint8)
        return state
