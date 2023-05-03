import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter
from srl.base.rl.model import IImageBlockConfig, IMLPBlockConfig
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import create_epsilon_list, inverse_rescaling, render_discrete_action
from srl.rl.models.dqn_image_block_config import DQNImageBlockConfig
from srl.rl.models.mlp_block_config import MLPBlockConfig
from srl.utils import common

"""
Paper
・Playing Atari with Deep Reinforcement Learning
https://arxiv.org/pdf/1312.5602.pdf

・Human-level control through deep reinforcement learning
https://www.nature.com/articles/nature14236


window_length          : -
Fixed Target Q-Network : o
Error clipping     : o
Experience Replay  : o
Frame skip         : -
Annealing e-greedy : o (config selection)
Reward clip        : o (config selection)
Image preprocessor : -

Other
    Double DQN               : o (config selection)
    Value function rescaling : o (config selection)
    invalid_actions          : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):
    test_epsilon: float = 0

    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    # Annealing e-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    exploration_steps: int = 0  # 0 : no Annealing

    discount: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 1000
    enable_reward_clip: bool = False

    # other
    enable_double_dqn: bool = True
    enable_rescale: bool = False

    # --- model
    image_block_config: IImageBlockConfig = field(default_factory=lambda: DQNImageBlockConfig())
    hidden_block_config: IMLPBlockConfig = field(default_factory=lambda: MLPBlockConfig(layer_sizes=(512,)))

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]

    # 論文のハイパーパラメーター
    def set_atari_config(self):
        self.batch_size = 32
        self.capacity = 1_000_000
        self.image_block_config = DQNImageBlockConfig()
        self.hidden_block_config = MLPBlockConfig(layer_sizes=(512,))
        self.target_model_update_interval = 10000
        self.discount = 0.99
        self.lr = 0.00025
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.exploration_steps = 1_000_000
        self.memory_warmup_size = 50_000
        self.enable_reward_clip = True
        self.enable_double_dqn = False
        self.enable_rescale = False

    # -------------------------------

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def getName(self) -> str:
        framework = self.get_use_framework()
        return f"DQN:{framework}"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(ExperienceReplayBuffer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(self.config.capacity)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter, ABC):
    @abstractmethod
    def get_q(self, state: np.ndarray, worker: "Worker") -> np.ndarray:
        raise NotImplementedError()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(CommonInterfaceParameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.step_epsilon = 0

        if self.config.exploration_steps > 0:
            self.initial_epsilon = self.config.initial_epsilon
            self.epsilon_step = (
                self.config.initial_epsilon - self.config.final_epsilon
            ) / self.config.exploration_steps
            self.final_epsilon = self.config.final_epsilon

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state

        if self.training:
            if self.config.exploration_steps > 0:
                # Annealing ε-greedy
                epsilon = self.initial_epsilon - self.step_epsilon * self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
            else:
                epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            q = self.parameter.get_q(state[np.newaxis, ...], self)[0]
            q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないので無視）
            self.action = int(np.argmax(q))

        return self.action, {"epsilon": epsilon}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        if not self.training:
            return {}
        self.step_epsilon += 1

        # reward clip
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        self.remote_memory.add(
            [
                self.state,
                next_state,
                np.identity(self.config.action_num, dtype=int)[self.action],
                reward,
                int(not done),
                next_invalid_actions,
            ]
        )

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        q = self.parameter.get_q(self.state[np.newaxis, ...], self)[0]
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(None, maxa, env, _render_sub)
