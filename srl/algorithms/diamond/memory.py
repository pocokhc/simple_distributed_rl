from typing import Any

from srl.base.rl.memory import RLMemory
from srl.rl.memories.episode_replay_buffer import EpisodeReplayBuffer

from .config import Config


class Memory(RLMemory[Config]):
    def setup(self):
        batch_length = self.config.burnin + self.config.horizon
        self.memory = EpisodeReplayBuffer(
            self.config.batch_size,
            self.config.memory.capacity,
            self.config.memory.warmup_size,
            self.config.memory.compress,
            self.config.memory.compress_level,
            suffix_size=batch_length,
        )

        self.add = self.memory.add
        self.register_worker_func_custom(self.memory.add, self.memory.serialize)
        self.register_trainer_recv_func(self.sample_diff)
        self.register_trainer_recv_func(self.sample_rewend)
        self.register_trainer_recv_func(self.sample_actor_critic)

    def length(self):
        return self.memory.length()

    def sample_diff(self):
        return self.memory.sample(
            suffix_size=self.config.denoiser_cfg.num_steps_conditioning,
            skip_tail=self.config.horizon,
        )

    def sample_rewend(self):
        return self.memory.sample(suffix_size=self.config.burnin + self.config.horizon)

    def sample_actor_critic(self):
        return self.memory.sample(
            suffix_size=self.config.denoiser_cfg.num_steps_conditioning,
            skip_tail=self.config.horizon,
        )

    def call_backup(self, **kwargs):
        return self.memory.call_backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.call_restore(data)
