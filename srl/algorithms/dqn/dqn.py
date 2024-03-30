import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, cast

import numpy as np

from srl.base.define import InfoType, RLBaseTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.worker import RLWorker
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
from srl.rl.functions.common import create_epsilon_list, inverse_rescaling, rescaling
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
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
class Config(
    RLConfig[DiscreteSpace, BoxSpace],
    RLConfigComponentExperienceReplayBuffer,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentExperienceReplayBuffer`>
    <:ref:`RLConfigComponentFramework`>
    """

    #: ε-greedy parameter for Test
    test_epsilon: float = 0

    #: Learning rate during distributed learning
    #: :math:`\epsilon_i = \epsilon^{1 + \frac{i}{N-1} \alpha}`
    actor_epsilon: float = 0.4
    #: Look actor_epsilon
    actor_alpha: float = 7.0

    #: <:ref:`scheduler`> ε-greedy parameter for Train
    epsilon: Union[float, SchedulerConfig] = 0.1
    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.001

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

    #: <:ref:`MLPBlock`> hidden layer
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    def __post_init__(self):
        super().__post_init__()

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        e = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]
        self.epsilon = e

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
        self.epsilon = self.create_scheduler().set_linear(1_000_000, 1.0, 0.1)

        self.enable_reward_clip = True
        self.enable_double_dqn = False
        self.enable_rescale = False

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE

    def get_framework(self) -> str:
        return self.create_framework_str()

    def get_name(self) -> str:
        return f"DQN:{self.get_framework()}"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()

    def get_changeable_parameters(self) -> List[str]:
        return ["test_epsilon"]

    def get_info_types(self) -> dict:
        return {
            "loss": {"data": "ave"},
            "sync": {"type": int, "data": "last"},
            "epsilon": {"data": "last"},
            "lr": {"data": "last"},
        }


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter[Config], ABC):
    @abstractmethod
    def create_batch_data(self, state):
        raise NotImplementedError()

    @abstractmethod
    def predict_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_target_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    def calc_target_q(
        self,
        batch_size,
        n_state,
        reward,
        done,
        next_invalid_actions,
    ):
        # ここの計算はtfで計算するよりnpで計算したほうが早い

        n_inv_act_idx1, n_inv_act_idx2 = helper.create_fancy_index_for_invalid_actions(next_invalid_actions)

        n_q_target = self.predict_target_q(n_state)

        # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
        if self.config.enable_double_dqn:
            n_q = self.predict_q(n_state)
            n_q[n_inv_act_idx1, n_inv_act_idx2] = np.min(n_q)
            n_act_idx = np.argmax(n_q, axis=1)
            maxq = n_q_target[np.arange(batch_size), n_act_idx]
        else:
            n_q_target[n_inv_act_idx1, n_inv_act_idx2] = np.min(n_q_target)
            maxq = np.max(n_q_target, axis=1)

        if self.config.enable_rescale:
            maxq = inverse_rescaling(maxq)

        # --- Q値を計算
        target_q = reward + done * self.config.discount * maxq

        if self.config.enable_rescale:
            target_q = rescaling(target_q)

        return target_q.astype(np.float32)


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, CommonInterfaceParameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

    def on_reset(self, worker) -> InfoType:
        return {}

    def policy(self, worker) -> Tuple[int, InfoType]:
        invalid_actions = worker.get_invalid_actions()

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice([a for a in range(self.action_space.n) if a not in invalid_actions])
            self.q = None
        else:
            state = self.parameter.create_batch_data(worker.state)
            self.q = self.parameter.predict_q(state)[0]
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(self.q))

        return action, {"epsilon": epsilon}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}

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
            done,
            next_invalid_actions,
        ]
        """
        action: int = cast(int, worker.prev_action)
        # memory.add に直接入れるとなぜか遅くなる
        batch = [
            worker.prev_state,
            worker.state,
            np.identity(self.action_space.n, dtype=int)[action],
            reward,
            int(not worker.terminated),
            worker.get_invalid_actions(),
        ]
        self.memory.add(batch)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            state = self.parameter.create_batch_data(worker.prev_state)
            q = self.parameter.predict_q(state)[0]
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        helper.render_discrete_action(int(maxa), self.action_space.n, worker.env, _render_sub)
