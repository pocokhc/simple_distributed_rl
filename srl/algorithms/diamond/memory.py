from typing import Any

from srl.base.rl.memory import RLMemory
from srl.rl.memories.episode_replay_buffer import EpisodeReplayBuffer

from .config import Config


class Memory(RLMemory[Config]):
    def setup(self):
        batch_length = self.config.burnin + self.config.horizon + 1
        self.memory = EpisodeReplayBuffer(self.config.memory, self.config.batch_size, batch_length)

        self.add = self.memory.add
        self.register_worker_func(self.memory.add, self.memory.serialize)
        self.register_trainer_recv_func(self.sample_diff)
        self.register_trainer_recv_func(self.sample_rewend)
        self.register_trainer_recv_func(self.sample_actor_critic)

    def length(self):
        return self.memory.length()

    def sample_diff(self):
        return self.memory.sample(
            batch_length=self.config.denoiser_cfg.num_steps_conditioning + 1,
            skip_tail=self.config.horizon,
        )

    def sample_rewend(self):
        return self.memory.sample(batch_length=self.config.burnin + self.config.horizon + 1)

    def sample_actor_critic(self):
        return self.memory.sample(
            batch_length=self.config.denoiser_cfg.num_steps_conditioning + 1,
            skip_tail=self.config.horizon,
        )

    def call_backup(self, **kwargs):
        return self.memory.call_backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.call_restore(data)
