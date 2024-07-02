import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.rl import functions as funcs
from srl.rl.functions import create_epsilon_list, inverse_rescaling, rescaling
from srl.rl.memories.priority_experience_replay import (
    PriorityExperienceReplay,
    RLConfigComponentPriorityExperienceReplay,
)
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

"""
・Paper
Rainbow: https://arxiv.org/abs/1710.02298
Double DQN: https://arxiv.org/abs/1509.06461
Priority Experience Replay: https://arxiv.org/abs/1511.05952
Dueling Network: https://arxiv.org/abs/1511.06581
Multi-Step learning: https://arxiv.org/abs/1703.01327
Retrace: https://arxiv.org/abs/1606.02647
Noisy Network: https://arxiv.org/abs/1706.10295
Categorical DQN: https://arxiv.org/abs/1707.06887

DQN
    window_length          : -
    Fixed Target Q-Network : o
    Error clipping     : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : o (config selection)
    Reward clip        : o (config selection)
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning(retrace): o (config selection)
    Noisy Network               : o (config selection)
    Categorical DQN             : x

Other
    Value function rescaling : o (config selection)
    invalid_actions : o

"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(
    RLConfig,
    RLConfigComponentPriorityExperienceReplay,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentPriorityExperienceReplay`>
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
    #: Learning rate
    lr: Union[float, SchedulerConfig] = 0.001

    #: <:ref:`MLPBlock`> hidden layer
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1000
    #: If True, clip the reward to three types [-1,0,1]
    enable_reward_clip: bool = False

    #: enable DoubleDQN
    enable_double_dqn: bool = True
    #: noisy dense
    enable_noisy_dense: bool = False
    #: enable rescaling
    enable_rescale: bool = False

    #: Multi-step learning
    multisteps: int = 3
    #: retrace parameter h
    retrace_h: float = 1.0

    dummy_state_val: float = 0

    def __post_init__(self):
        super().__post_init__()
        self.set_proportional_memory()
        self.hidden_block.set_dueling_network((512,))

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        e = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]
        self.epsilon = e

    def set_atari_config(self):
        # Annealing e-greedy
        self.epsilon = self.create_scheduler().set_linear(1_000_000, 1.0, 0.1)

        # model
        self.input_image_block.set_dqn_block()
        self.hidden_block.set_dueling_network((512,), dueling_type="average")
        self.enable_double_dqn = True

        self.discount = 0.99
        self.lr = 0.0000625
        self.batch_size = 32
        self.target_model_update_interval = 32000
        self.enable_reward_clip = True

        # memory
        self.memory_warmup_size = 80_000
        self.memory_capacity = 1_000_000
        self.set_proportional_memory(
            alpha=0.5,
            beta_initial=0.4,
            beta_steps=1_000_000,
        )

        # Multi-step learning
        self.multisteps = 3
        self.retrace_h = 1.0

        # noisy dense
        self.enable_noisy_dense = True

        # other
        self.enable_rescale = False

    def get_processors(self) -> List[Optional[RLProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_framework(self) -> str:
        return self.create_framework_str()

    def get_name(self) -> str:
        if self.multisteps == 1:
            return f"Rainbow_no_multisteps:{self.get_framework()}"
        else:
            return f"Rainbow:{self.get_framework()}"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()
        assert self.multisteps > 0


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(PriorityExperienceReplay):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter[Config], ABC):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.multi_discounts = np.array([self.config.discount**n for n in range(self.config.multisteps)])

    @abstractmethod
    def pred_single_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def pred_batch_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def pred_batch_target_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    def calc_target_q(self, batchs, training: bool):
        batch_size = len(batchs)
        multi_discounts = np.tile(self.multi_discounts, (batch_size, 1))

        # (batch, multistep, shape)
        states_list, onehot_actions_list, rewards, dones, _ = zip(*batchs)
        states_list = np.asarray(states_list)
        onehot_actions_list = np.asarray(onehot_actions_list, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # 1step目はretraceで使わない、retraceで使うのは 2step以降
        states = states_list[:, 0, :]
        n_states = states_list[:, 1:, :]
        onehot_actions = onehot_actions_list[:, 0, :]
        n_onehot_actions = onehot_actions_list[:, 1:, :]

        """
        - actionQ (online) 1step ～ N step
        - nextQ
          double_dqn
            (online) 1step ～ N+1 step
            (target) 1step ～ N+1 step
          no double_dqn
            (online) no use
            (target) 1step ～ N+1 step
        """
        if self.config.enable_double_dqn:
            online_states = n_states
            target_states = n_states
        else:
            online_states = n_states[:, :-1, :]
            target_states = n_states

        # (batch, multistep, shape) -> (batch * multistep, shape)
        online_shape1 = online_states.shape[1]
        online_states = np.reshape(online_states, (batch_size * online_shape1,) + online_states.shape[2:])
        target_states = np.reshape(target_states, (batch_size * self.config.multisteps,) + target_states.shape[2:])

        q_online = self.pred_batch_q(online_states)
        q_target = self.pred_batch_target_q(target_states)

        # (batch * multistep, shape) -> (batch, multistep, shape)
        q_online = np.reshape(q_online, (batch_size, online_shape1) + q_online.shape[1:])
        q_target = np.reshape(q_target, (batch_size, self.config.multisteps) + q_target.shape[1:])

        # --- action Q
        q = np.sum(q_online[:, : self.config.multisteps - 1, :] * n_onehot_actions, axis=2)
        # 1step目は学習側で計算するので0をいれる
        q = np.insert(q, 0, 0, axis=1)

        # --- calc TD error
        # ファンシーインデックス
        idx1 = [i for i, b in enumerate(batchs) for e in b[4] for e2 in e]
        idx2 = [i for b in batchs for i, e in enumerate(b[4]) for e2 in e]
        idx3 = [e2 for b in batchs for e in b[4] for e2 in e]
        if self.config.enable_double_dqn:
            q_online[idx1, idx2, idx3] = -np.inf
            n_act_idx = np.argmax(q_online, axis=2)
        else:
            q_target[idx1, idx2, idx3] = -np.inf
            n_act_idx = np.argmax(q_target, axis=2)
        maxq = np.take_along_axis(q_target, np.expand_dims(n_act_idx, axis=2), axis=2)
        maxq = np.squeeze(maxq, axis=2)

        if self.config.enable_rescale:
            maxq = inverse_rescaling(maxq)

        gains = rewards + dones * self.config.discount * maxq

        if self.config.enable_rescale:
            gains = rescaling(gains)

        td_errors = gains - q

        # --- calc retrace
        # 各batchで最大のアクションを選んでるかどうか
        # greedyな方策なので、最大アクションなら確率1.0 #[1]
        pi_probs = np.argmax(n_onehot_actions, axis=2) == n_act_idx[:, 1:]

        #  (batch, multistep, shape) -> (multistep, batch, shape)
        # mu_probs = np.transpose(mu_probs, (1, 0)) #[1]
        pi_probs = np.transpose(pi_probs, (1, 0))

        retrace_list = [np.ones((batch_size,))]  # 0stepはretraceなし
        retrace = np.ones((batch_size,))
        for n in range(self.config.multisteps - 1):
            # #[1]
            # pi_probs は 0 or 1 で mu_probs は1以下なので必ず 0 or 1 になる
            # retrace *= self.config.retrace_h * np.minimum(1, pi_probs[n] / mu_probs[n])
            retrace *= self.config.retrace_h * pi_probs[n]
            retrace_list.append(retrace.copy())

        # (multistep, batch, shape) ->  (batch, multistep, shape)
        retrace_list = np.asarray(retrace_list).transpose((1, 0))

        target_q = np.sum(td_errors * multi_discounts * retrace_list, axis=1, dtype=np.float32)

        if training:
            return target_q, states, onehot_actions
        else:
            return target_q


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, CommonInterfaceParameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.dummy_state = np.full(self.config.observation_space.shape, self.config.dummy_state_val, dtype=np.float32)
        self.onehot_arr = np.identity(self.config.action_space.n, dtype=int)

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

    def on_reset(self, worker):
        self._recent_states = [self.dummy_state for _ in range(self.config.multisteps + 1)]
        self._recent_actions = [
            self.onehot_arr[random.randint(0, self.config.action_space.n - 1)] for _ in range(self.config.multisteps)
        ]
        # self._recent_probs = [1.0 / self.config.action_space.n for _ in range(self.config.multisteps)] #[1]
        self._recent_rewards = [0.0 for _ in range(self.config.multisteps)]
        self._recent_done = [1 for _ in range(self.config.multisteps)]
        self._recent_invalid_actions = [[] for _ in range(self.config.multisteps)]

        self._recent_states.pop(0)
        self._recent_states.append(worker.state)

    def policy(self, worker) -> int:
        self.state = worker.state
        invalid_actions = worker.get_invalid_actions()

        if self.config.enable_noisy_dense:
            self.q = self.parameter.pred_single_q(self.state)
            self.q[invalid_actions] = -np.inf
            self.action = int(np.argmax(self.q))
            # self.prob = 1.0 #[1]
            return self.action

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        # valid_action_num = self.config.action_space.n - len(invalid_actions)
        if random.random() < epsilon:
            self.action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
            self.q = None
            # self.prob = epsilon / valid_action_num #[1]
        else:
            self.q = self.parameter.pred_single_q(self.state)
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            self.action = int(np.argmax(self.q))
            # self.prob = epsilon / valid_action_num + (1 - epsilon) #[1]

        self.info["epsilon"] = epsilon
        return self.action

    def on_step(self, worker):
        reward = worker.reward
        self._recent_states.pop(0)
        self._recent_states.append(worker.state)

        if not self.training:
            return

        # reward clip
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        self._recent_actions.pop(0)
        self._recent_actions.append(self.onehot_arr[self.action])
        # self._recent_probs.pop(0)
        # self._recent_probs.append(self.prob) #[1]
        self._recent_rewards.pop(0)
        self._recent_rewards.append(reward)
        self._recent_done.pop(0)
        self._recent_done.append(int(not worker.terminated))
        self._recent_invalid_actions.pop(0)
        self._recent_invalid_actions.append(worker.get_invalid_actions())
        priority = self._add_memory(None)

        if worker.done:
            # 残りstepも追加
            for _ in range(len(self._recent_rewards) - 1):
                self._recent_states.pop(0)
                self._recent_states.append(self.dummy_state)
                self._recent_actions.pop(0)
                self._recent_actions.append(self.onehot_arr[random.randint(0, self.config.action_space.n - 1)])
                # self._recent_probs.pop(0)
                # self._recent_probs.append(1.0) #[1]
                self._recent_rewards.pop(0)
                self._recent_rewards.append(0.0)
                self._recent_done.pop(0)
                self._recent_done.append(0)
                self._recent_invalid_actions.pop(0)
                self._recent_invalid_actions.append([])
                self._add_memory(priority)

    def _add_memory(self, priority):
        """
        [
            states,
            onehot_actions,
            # probs,
            rewards,
            dones,
            invalid_actions,
        ]
        """
        batch = [
            self._recent_states[:],
            self._recent_actions[:],
            # self._recent_probs[:], #[1]
            self._recent_rewards[:],
            self._recent_done[:],
            self._recent_invalid_actions[:],
        ]

        if priority is None:
            if not self.distributed:
                priority = None
            elif not self.config.requires_priority():
                priority = None
            else:
                if self.q is None:
                    self.q = self.parameter.pred_single_q(self.state)
                select_q = self.q[self.action]
                target_q = self.parameter.calc_target_q([batch], training=False)[0]
                priority = abs(target_q - select_q)

        self.memory.add(batch, priority)
        return priority

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.pred_single_q(self.state)
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
