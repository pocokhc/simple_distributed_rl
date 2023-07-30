from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from srl.base.rl.base import RLRemoteMemory

from .priority_memories.imemory import IPriorityMemory


@dataclass
class _PriorityExperienceReplayConfig:
    capacity: int = 100_000
    _name: str = field(init=False, default="ReplayMemory")
    _kwargs: dict = field(init=False, default_factory=dict)

    enable_best_episode_memory: bool = field(init=False, default=False)
    best_episode_memory: Optional["_PriorityExperienceReplayConfig"] = field(init=False, default=None)
    best_episode_ratio: float = 1.0 / 256.0
    best_episode_has_reward_equal: bool = True

    enable_demo_memory: bool = field(init=False, default=False)
    demo_memory: Optional["_PriorityExperienceReplayConfig"] = field(init=False, default=None)
    demo_memory_playing: bool = True
    demo_memory_ratio: float = 1.0 / 256.0

    def set_replay_memory(self):
        self._name = "ReplayMemory"

    def set_proportional_memory(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
        has_duplicate: bool = True,
        epsilon: float = 0.0001,
    ):
        self._name = "ProportionalMemory"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
            has_duplicate=has_duplicate,
            epsilon=epsilon,
        )

    def set_rankbase_memory(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self._name = "RankBaseMemory"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )

    def set_rankbase_memory_linear(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self._name = "RankBaseMemoryLinear"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )

    def set_best_episode_memory(
        self,
        enable_best_episode_memory: bool = True,
        ratio: float = 1.0 / 256.0,  # 混ぜる割合
        has_reward_equal: bool = True,
    ):
        self.enable_best_episode_memory = enable_best_episode_memory
        if not enable_best_episode_memory:
            self.best_episode_memory = None
            return
        self.best_episode_memory = _PriorityExperienceReplayConfig()
        self.best_episode_ratio = ratio
        self.best_episode_has_reward_equal = has_reward_equal

    def set_demo_memory(
        self,
        enable_demo_memory: bool = True,
        playing: bool = False,
        ratio: float = 1.0 / 256.0,  # 混ぜる割合
    ):
        self.enable_demo_memory = enable_demo_memory
        if not enable_demo_memory:
            self.demo_memory = None
            return
        self.demo_memory = _PriorityExperienceReplayConfig()
        self.demo_memory_playing = playing
        self.demo_memory_ratio = ratio

    def set_original_memory(self):
        raise NotImplementedError("TODO")

    # ---------------------------

    def create_memory(self) -> IPriorityMemory:
        if self._name == "ReplayMemory":
            from .priority_memories.replay_memory import ReplayMemory

            memory = ReplayMemory(self.capacity)
        elif self._name == "ProportionalMemory":
            from .priority_memories.proportional_memory import ProportionalMemory

            memory = ProportionalMemory(self.capacity, **self._kwargs)

        elif self._name == "RankBaseMemory":
            from .priority_memories.rankbase_memory import RankBaseMemory

            memory = RankBaseMemory(self.capacity, **self._kwargs)

        elif self._name == "RankBaseMemoryLinear":
            from .priority_memories.rankbase_memory_linear import RankBaseMemoryLinear

            memory = RankBaseMemoryLinear(self.capacity, **self._kwargs)
        else:
            raise ValueError(self._name)

        if self.enable_demo_memory:
            assert self.demo_memory is not None
            from .priority_memories.demo_memory import DemoMemory

            memory = DemoMemory(
                main_memory=memory,
                demo_memory=self.demo_memory.create_memory(),
                playing=self.demo_memory_playing,
                ratio=self.demo_memory_ratio,
            )
        elif self.enable_best_episode_memory:
            assert self.best_episode_memory is not None
            from .priority_memories.best_episode_memory import BestEpisodeMemory

            memory = BestEpisodeMemory(
                main_memory=memory,
                best_memory=self.best_episode_memory.create_memory(),
                ratio=self.best_episode_ratio,
                has_reward_equal=self.best_episode_has_reward_equal,
            )

        return memory

    def requires_priority(self) -> bool:
        if self._name == "ReplayMemory":
            return False
        elif self._name == "ProportionalMemory":
            return True
        elif self._name == "RankBaseMemory":
            return True
        elif self._name == "RankBaseMemoryLinear":
            return True
        return False


@dataclass
class PriorityExperienceReplayConfig:
    memory: _PriorityExperienceReplayConfig = field(
        init=False, default_factory=lambda: _PriorityExperienceReplayConfig()
    )


class PriorityExperienceReplay(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: PriorityExperienceReplayConfig = self.config
        self.memory = self.config.memory.create_memory()

    def length(self) -> int:
        return len(self.memory)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    # ---------------------------

    def add(self, batch: Any, td_error: Optional[float] = None):
        self.memory.add(batch, td_error)

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        return self.memory.sample(batch_size, step)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        self.memory.update(indices, batchs, td_errors)

    def on_step(self, reward: float, done: bool) -> None:
        self.memory.on_step(reward, done)
