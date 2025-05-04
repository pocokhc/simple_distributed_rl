import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

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
    Double DQN                 : o (config selection)
    Priority Experience Replay : o (config selection)
    Value function rescaling   : o (config selection)
    invalid_actions            : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, RLConfigComponentFramework):
    """
    <:ref:`RLConfigComponentFramework`>
    """

    #: ε-greedy parameter for Test
    test_epsilon: float = 0

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: ε-greedy parameter for Train
    epsilon: float = 0.1
    #: <:ref:`SchedulerConfig`>
    epsilon_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1000
    #: If True, clip the reward to three types [-1,0,1]
    enable_reward_clip: bool = False

    #: enable DoubleDQN
    enable_double_dqn: bool = True
    #: enable rescaling
    enable_rescale: bool = False

    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    #: <:ref:`MLPBlockConfig`> hidden layer
    hidden_block: MLPBlockConfig = field(default_factory=lambda: MLPBlockConfig())

    def set_atari_config(self):
        """Set the Atari parameters written in the paper."""
        self.batch_size = 32
        self.memory.capacity = 1_000_000
        self.memory.warmup_size = 50_000
        self.input_image_block.set_dqn_block()
        self.hidden_block.set((512,))
        self.target_model_update_interval = 10000
        self.discount = 0.99
        self.lr = 0.00025
        self.epsilon_scheduler.set_linear(1.0, 0.1, 1_000_000)
        self.enable_reward_clip = True
        self.enable_double_dqn = False
        self.enable_rescale = False

    def get_name(self) -> str:
        return "DQN"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []

    def get_framework(self) -> str:
        return RLConfigComponentFramework.get_framework(self)


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(RLPriorityReplayBuffer):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter[Config], ABC):
    @abstractmethod
    def pred_single_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def pred_batch_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def pred_batch_target_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    def calc_target_q(
        self,
        batch_size,
        n_state,
        reward,
        undone,
        next_invalid_actions,
    ):
        # ここの計算はtfで計算するよりnpで計算したほうが早い

        n_inv_act_idx1, n_inv_act_idx2 = funcs.create_fancy_index_for_invalid_actions(next_invalid_actions)

        n_q_target = self.pred_batch_target_q(n_state)

        # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
        if self.config.enable_double_dqn:
            n_q = self.pred_batch_q(n_state)
            n_q[n_inv_act_idx1, n_inv_act_idx2] = np.min(n_q)
            n_act_idx = np.argmax(n_q, axis=1)
            maxq = n_q_target[np.arange(batch_size), n_act_idx]
        else:
            n_q_target[n_inv_act_idx1, n_inv_act_idx2] = np.min(n_q_target)
            maxq = np.max(n_q_target, axis=1)

        if self.config.enable_rescale:
            maxq = funcs.inverse_rescaling(maxq)

        # --- Q値を計算
        target_q = reward + undone * self.config.discount * maxq

        if self.config.enable_rescale:
            target_q = funcs.rescaling(target_q)

        return target_q.astype(self.config.get_dtype("np"))


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, CommonInterfaceParameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)

    def on_teardown(self, worker) -> None:
        pass

    def on_reset(self, worker):
        pass

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions

        if self.training:
            epsilon = self.epsilon_sch.update(self.step_in_training).to_float()
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
        else:
            q = self.parameter.pred_single_q(worker.state)
            q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        self.info["epsilon"] = epsilon
        return action

    def on_step(self, worker):
        if not self.training:
            return

        # reward clip
        reward = worker.reward
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        """
        [
            state,
            n_state,
            onehot_action,
            reward,
            undone,
            next_invalid_actions,
        ]
        """
        self.memory.add(
            [
                worker.state,
                worker.next_state,
                worker.get_onehot_action(),
                reward,
                int(not worker.terminated),
                worker.next_invalid_actions,
            ]
        )

    def render_terminal(self, worker, **kwargs):
        # policy -> render -> env.step -> on_step
        q = self.parameter.pred_single_q(worker.state)
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = funcs.inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        worker.print_discrete_action_info(int(maxa), _render_sub)
