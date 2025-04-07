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
from srl.rl.functions import create_epsilon_list, inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
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

    #: Learning rate during distributed learning
    #: :math:`\epsilon_i = \epsilon^{1 + \frac{i}{N-1} \alpha}`
    actor_epsilon: float = 0.4
    #: Look actor_epsilon
    actor_alpha: float = 7.0

    #: ε-greedy parameter for Train
    epsilon: float = 0.1
    #: <:ref:`SchedulerConfig`>
    epsilon_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    #: <:ref:`DuelingNetworkConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

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

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(
            actor_num,
            epsilon=self.actor_epsilon,
            alpha=self.actor_alpha,
        )[actor_id]

    def set_atari_config(self):
        # Annealing e-greedy
        self.epsilon_scheduler.set_linear(1.0, 0.1, 1_000_000)

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
        self.memory.warmup_size = 80_000
        self.memory.capacity = 1_000_000
        self.memory.set_proportional(
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

    def get_name(self) -> str:
        if self.multisteps == 1:
            return "Rainbow_no_multisteps"
        else:
            return "Rainbow"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []

    def get_framework(self) -> str:
        return RLConfigComponentFramework.get_framework(self)

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.multisteps > 0):
            raise ValueError(f"assert {self.multisteps} > 0")


class Memory(RLPriorityReplayBuffer):
    pass


class CommonInterfaceParameter(RLParameter[Config], ABC):
    def setup(self):
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

    def calc_target_q(self, batches, training: bool):
        batch_size = len(batches)
        multi_discounts = np.tile(self.multi_discounts, (batch_size, 1))

        # (batch, multistep, shape)
        states_list, onehot_actions_list, rewards, dones, _ = zip(*batches)
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
        idx1 = [i for i, b in enumerate(batches) for e in b[4] for e2 in e]
        idx2 = [i for b in batches for i, e in enumerate(b[4]) for e2 in e]
        idx3 = [e2 for b in batches for e in b[4] for e2 in e]
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

        gains = rewards + (1 - dones) * self.config.discount * maxq

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


class Worker(RLWorker[Config, CommonInterfaceParameter, Memory]):
    def on_setup(self, worker, context):
        self.np_dtype = self.config.get_dtype("np")
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)

        # tracking機能を有効化
        worker.enable_tracking(max_size=self.config.multisteps)

    def on_reset(self, worker):
        for _ in range(self.config.multisteps - 1):
            worker.add_dummy_step(
                tracking_data={
                    "onehot_action": worker.get_onehot_action(random.randint(0, self.config.action_space.n - 1)),
                    "clip_reward": 0,
                    # "prob": 1.0,  # [1]
                },
                is_reset=True,
            )

    def policy(self, worker) -> int:
        state = worker.state
        invalid_actions = worker.invalid_actions

        if self.config.enable_noisy_dense:
            self.q = self.parameter.pred_single_q(state)
            self.q[invalid_actions] = -np.inf
            # self.prob = 1.0  #[1]
            return int(np.argmax(self.q))

        if self.training:
            epsilon = self.epsilon_sch.update(self.total_step).to_float()
        else:
            epsilon = self.config.test_epsilon

        # valid_action_num = self.config.action_space.n - len(invalid_actions)  #[1]
        if random.random() < epsilon:
            action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
            self.q = None
            # self.prob = epsilon / valid_action_num  #[1]
        else:
            self.q = self.parameter.pred_single_q(state)
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            action = int(np.argmax(self.q))
            # self.prob = epsilon / valid_action_num + (1 - epsilon) #[1]

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

        worker.add_tracking(
            {
                "onehot_action": worker.get_onehot_action(),
                "clip_reward": reward,
                # "prob": self.prob,  # [1]
            }
        )

        batch = [
            worker.get_tracking("state", self.config.multisteps + 1),
            worker.get_tracking("onehot_action", self.config.multisteps),
            worker.get_tracking("clip_reward", self.config.multisteps),
            worker.get_tracking("terminated", self.config.multisteps),
            worker.get_tracking("invalid_actions", self.config.multisteps),
            # worker.get_tracking_data("prob", self.config.multisteps),  #[1]
        ]

        if not self.distributed:
            priority = None
        elif not self.config.memory.requires_priority():
            priority = None
        else:
            if self.q is None:
                self.q = self.parameter.pred_single_q(worker.prev_state)
            select_q = self.q[worker.action]
            target_q = self.parameter.calc_target_q([batch], training=False)[0]
            priority = abs(target_q - select_q)

        self.memory.add(batch, priority)

        if worker.done:
            # 残りstepも追加
            for _ in range(self.config.multisteps - 1):
                worker.add_dummy_step(
                    terminated=True,
                    tracking_data={
                        "onehot_action": worker.get_onehot_action(random.randint(0, self.config.action_space.n - 1)),
                        "clip_reward": 0,
                        # "prob": 1.0,  # [1]
                    },
                )
                self.memory.add(
                    [
                        worker.get_tracking("state", self.config.multisteps + 1),
                        worker.get_tracking("onehot_action", self.config.multisteps),
                        worker.get_tracking("clip_reward", self.config.multisteps),
                        worker.get_tracking("terminated", self.config.multisteps),
                        worker.get_tracking("invalid_actions", self.config.multisteps),
                        # worker.get_tracking_data("prob", self.config.multisteps),  #[1]
                    ],
                    priority,
                )

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.pred_single_q(worker.state)
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
