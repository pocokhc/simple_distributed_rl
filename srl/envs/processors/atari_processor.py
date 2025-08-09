from dataclasses import dataclass
from typing import List, Optional, Tuple

import ale_py  # include atari gym # noqa: F401
import numpy as np

from srl.base.define import SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.env.processor import EnvProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.functions import image_processor


@dataclass
class AtariProcessor(EnvProcessor):
    terminal_on_life_loss: bool = False
    resize: Optional[Tuple[int, int]] = None
    grayscale: bool = True
    binarize: bool = False

    def __post_init__(self):
        self.space_type = SpaceTypes.GRAY_2ch if self.grayscale else SpaceTypes.COLOR

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> SpaceBase:
        if self.resize is None:
            return prev_space
        return BoxSpace(self.resize, 0, 255, np.uint8, stype=self.space_type)

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
        if self.resize is None:
            return state
        state = image_processor(
            state,
            prev_space.stype,
            self.space_type,
            resize=self.resize,
        )
        if self.binarize:
            state = np.where(state > 127, 255, 0).astype(np.uint8)
        return state

    def remap_reset(self, env_run: EnvRun, **kwargs):
        self.lives = env_run.unwrapped.env.unwrapped.ale.lives()

    def remap_step(self, rewards: List[float], terminated: bool, truncated: bool, env_run: EnvRun, **kwargs):
        if self.terminal_on_life_loss:
            new_lives = env_run.unwrapped.env.unwrapped.ale.lives()
            if new_lives < self.lives:
                return rewards, True, truncated
            self.lives = new_lives
        return rewards, terminated, truncated


@dataclass
class AtariPongProcessor(EnvProcessor):
    resize: Tuple[int, int] = (64, 64)

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> SpaceBase:
        return BoxSpace(self.resize, 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch)

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
        state = image_processor(
            state,
            prev_space.stype,
            SpaceTypes.GRAY_2ch,
            trimming=(35, 10, 195, 150),  # (0, 0, 210, 160)
            resize=self.resize,
        )
        state = np.where(state > 127, 255, 0).astype(np.uint8)
        return state

    def remap_reset(self, **kwargs):
        self.point = 0

    def remap_step(self, rewards: List[float], terminated: bool, truncated: bool, env_run: EnvRun, **kwargs):
        if env_run.reward == 1:
            self.point += 1
        elif env_run.reward == -1:
            self.point += 1
        if self.point == 5:
            terminated = True
        return rewards, terminated, truncated

    def backup(self):
        return self.point

    def restore(self, dat):
        self.point = dat


@dataclass
class AtariBreakoutProcessor(EnvProcessor):
    terminal_on_life_loss: bool = True

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> SpaceBase:
        return BoxSpace((84, 84), 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch)

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
        state = image_processor(
            state,
            prev_space.stype,
            SpaceTypes.GRAY_2ch,
            trimming=(31, 7, 195, 153),  # (0, 0, 210, 160)
            resize=(84, 84),
        )
        state = np.where(state > 50, 255, 0).astype(np.uint8)
        return state

    def remap_reset(self, env_run: EnvRun, **kwargs):
        self.lives = env_run.unwrapped.env.unwrapped.ale.lives()

    def remap_step(self, rewards: List[float], terminated: bool, truncated: bool, env_run: EnvRun, **kwargs):
        if self.terminal_on_life_loss:
            new_lives = env_run.unwrapped.env.unwrapped.ale.lives()
            if new_lives < self.lives:
                return [-1], True, truncated
            self.lives = new_lives
        return rewards, terminated, truncated


@dataclass
class AtariFreewayProcessor(EnvProcessor):
    resize: Tuple[int, int] = (64, 64)

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> SpaceBase:
        return BoxSpace(self.resize, 0, 255, np.uint8, stype=SpaceTypes.GRAY_2ch)

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, **kwargs):
        state = image_processor(
            state,
            prev_space.stype,
            SpaceTypes.GRAY_2ch,
            trimming=(30, 0, 210 - 20, 160),  # (0, 0, 210, 160)
            resize=self.resize,
        )
        # state = np.where(state > 150, 255, 0).astype(np.uint8)
        return state

    def remap_reset(self, **kwargs):
        self.step = 0

    def remap_step(self, rewards: List[float], terminated: bool, truncated: bool, env_run: EnvRun, **kwargs):
        self.step += 1
        if self.step > 200:
            truncated = True
        return rewards, terminated, truncated

    def backup(self):
        return self.point

    def restore(self, dat):
        self.point = dat
