from dataclasses import dataclass, field

from srl.base.rl.memory import IPriorityMemory, IPriorityMemoryConfig


@dataclass
class ReplayMemoryConfig(IPriorityMemoryConfig):
    capacity: int = 100_000

    def get_capacity(self) -> int:
        return self.capacity

    def create_memory(self) -> IPriorityMemory:
        from .replay_memory import ReplayMemory

        return ReplayMemory(self.capacity)

    def is_replay_memory(self) -> bool:
        return True


@dataclass
class ProportionalMemoryConfig(IPriorityMemoryConfig):
    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000
    has_duplicate: bool = True
    epsilon: float = 0.0001

    def get_capacity(self) -> int:
        return self.capacity

    def create_memory(self) -> IPriorityMemory:
        from .proportional_memory import ProportionalMemory

        return ProportionalMemory(
            self.capacity,
            self.alpha,
            self.beta_initial,
            self.beta_steps,
            self.has_duplicate,
            self.epsilon,
        )

    def is_replay_memory(self) -> bool:
        return False


@dataclass
class RankBaseMemoryConfig(IPriorityMemoryConfig):
    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

    def get_capacity(self) -> int:
        return self.capacity

    def create_memory(self) -> IPriorityMemory:
        from .rankbase_memory import RankBaseMemory

        return RankBaseMemory(
            self.capacity,
            self.alpha,
            self.beta_initial,
            self.beta_steps,
        )

    def is_replay_memory(self) -> bool:
        return False


@dataclass
class RankBaseMemoryLinearConfig(IPriorityMemoryConfig):
    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

    def get_capacity(self) -> int:
        return self.capacity

    def create_memory(self) -> IPriorityMemory:
        from .rankbase_memory_linear import RankBaseMemoryLinear

        return RankBaseMemoryLinear(
            self.capacity,
            self.alpha,
            self.beta_initial,
            self.beta_steps,
        )

    def is_replay_memory(self) -> bool:
        return False


@dataclass
class BestEpisodeMemoryConfig(IPriorityMemoryConfig):
    ratio: float = 1.0 / 256.0  # 混ぜる割合
    has_reward_equal: bool = True
    memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())

    def get_capacity(self) -> int:
        return self.memory.get_capacity()

    def create_memory(self) -> IPriorityMemory:
        from .best_episode_memory import BestEpisodeMemory

        return BestEpisodeMemory(
            self.ratio,
            self.has_reward_equal,
            self.memory,
        )

    def is_replay_memory(self) -> bool:
        return self.memory.is_replay_memory()


@dataclass
class DemoMemoryConfig(IPriorityMemoryConfig):
    playing: bool = False
    ratio: float = 1.0 / 256.0  # 混ぜる割合
    memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())
    demo_memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())

    def get_capacity(self) -> int:
        return self.memory.get_capacity()

    def create_memory(self) -> IPriorityMemory:
        from .demo_memory import DemoMemory

        return DemoMemory(
            self.playing,
            self.ratio,
            self.memory,
            self.demo_memory,
        )

    def is_replay_memory(self) -> bool:
        return self.memory.is_replay_memory()
