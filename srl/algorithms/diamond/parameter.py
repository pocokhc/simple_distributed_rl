from typing import Any

from srl.base.rl.parameter import RLParameter

from .config import Config
from .model_actor_critic import ActorCritic
from .model_denoiser import Denoiser
from .model_reward_end import RewardEndModel
from .model_sampler import DiffusionSampler


class Parameter(RLParameter[Config]):
    def setup(self):
        self.denoiser = Denoiser(self.config.observation_space.shape, self.config.action_space.n, self.config.denoiser_cfg)
        self.sampler = DiffusionSampler(self.denoiser, self.config.sampler_cfg, self.config.get_dtype("tf"))
        self.reward_end_model = RewardEndModel(self.config.observation_space.shape, self.config.action_space.n, self.config.reward_end_cfg)
        self.actor_critic = ActorCritic(self.config.observation_space.shape, self.config.action_space.n, self.config.actor_critic_cfg)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.denoiser.set_weights(data[0])
        self.reward_end_model.set_weights(data[1])
        self.actor_critic.set_weights(data[2])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.denoiser.get_weights(),
            self.reward_end_model.get_weights(),
            self.actor_critic.get_weights(),
        ]

    def summary(self, **kwargs):
        self.denoiser.summary(expand_nested=False)
        self.reward_end_model.summary(expand_nested=False)
        self.actor_critic.summary(expand_nested=False)
