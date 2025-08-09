import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.space import SpaceBase
from srl.rl.functions import create_epsilon_list, inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.input_block import InputBlockConfig
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
    epsilon_scheduler: SchedulerConfig = field(init=False, default_factory=lambda: SchedulerConfig())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(init=False, default_factory=lambda: LRSchedulerConfig())

    #: <:ref:`InputBlockConfig`>
    input_block: InputBlockConfig = field(init=False, default_factory=lambda: InputBlockConfig())
    #: <:ref:`DuelingNetworkConfig`> hidden+out layer
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
        self.input_block.image.set_dqn_block()
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
        return self.input_block.get_processors(prev_observation_space)

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
        self.np_dtype = self.config.get_dtype("np")
        self.multi_discounts = np.array([self.config.discount**n for n in range(self.config.multisteps)], dtype=self.np_dtype)

    @abstractmethod
    def pred_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def pred_target_q(self, state) -> np.ndarray:
        raise NotImplementedError()

    def calc_target_q(self, batches):
        batch_size = len(batches)
        multi_discounts = np.tile(self.multi_discounts, (batch_size, 1))

        # (batch, multistep, shape)
        state_list = np.asarray([[b[0] for b in steps] for steps in batches])
        # act,r,dは[1:]
        action_list = np.asarray([[b[1] for b in steps[1:]] for steps in batches], dtype=self.np_dtype)
        reward = np.array([[b[2] for b in steps[1:]] for steps in batches], dtype=self.np_dtype)
        done = np.array([[b[3] for b in steps[1:]] for steps in batches], dtype=self.np_dtype)

        # 1step目はretraceで使わない、retraceで使うのは 2step以降
        state = state_list[:, 0, :]
        n_state = state_list[:, 1:, :]
        action = action_list[:, 0, :]
        n_action = action_list[:, 1:, :]

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
            online_state = n_state
            target_state = n_state
        else:
            online_state = n_state[:, :-1, :]
            target_state = n_state

        # (batch, multistep, shape) -> (batch * multistep, shape)
        online_shape1 = online_state.shape[1]
        online_state = np.reshape(online_state, (batch_size * online_shape1,) + online_state.shape[2:])
        target_state = np.reshape(target_state, (batch_size * self.config.multisteps,) + target_state.shape[2:])

        q_online = self.pred_q(online_state)
        q_target = self.pred_target_q(target_state)

        # (batch * multistep, shape) -> (batch, multistep, shape)
        q_online = np.reshape(q_online, (batch_size, online_shape1) + q_online.shape[1:])
        q_target = np.reshape(q_target, (batch_size, self.config.multisteps) + q_target.shape[1:])

        # --- action Q
        q = np.sum(q_online[:, : self.config.multisteps - 1, :] * n_action, axis=2)
        # 1step目は学習側で計算するので0をいれる
        q = np.insert(q, 0, 0, axis=1)

        # --- calc TD error
        # invalid_actionsは[:-1]
        # ファンシーインデックス
        # idx1: batch index
        # idx2: step index
        # idx3: invalid action
        idx1 = [i for i, steps in enumerate(batches) for b in steps[1:] for e in b[4]]
        idx2 = [i for steps in batches for i, b in enumerate(steps[1:]) for e in b[4]]
        idx3 = [e for steps in batches for b in steps[1:] for e in b[4]]
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

        gains = reward + (1 - done) * self.config.discount * maxq

        if self.config.enable_rescale:
            gains = rescaling(gains)

        td_errors = gains - q

        # --- calc retrace
        # 各batchで最大のアクションを選んでるかどうか
        # greedyな方策なので、最大アクションなら確率1.0 #[1]
        pi_probs = np.argmax(n_action, axis=2) == n_act_idx[:, 1:]

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

        target_q = np.sum(td_errors * multi_discounts * retrace_list, axis=1, dtype=self.np_dtype)

        return target_q, state, action


class Worker(RLWorker[Config, CommonInterfaceParameter, Memory]):
    def on_setup(self, worker, context):
        self.np_dtype = self.config.get_dtype("np")
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)

        # tracking機能を利用
        worker.set_tracking_max_size(self.config.multisteps + 1)

    def on_reset(self, worker):
        worker.add_tracking({"state": worker.state})

    def policy(self, worker) -> int:
        state = worker.state
        invalid_actions = worker.invalid_actions

        if self.config.enable_noisy_dense:
            self.q = self.parameter.pred_q(state[np.newaxis, ...])[0]
            self.q[invalid_actions] = -np.inf
            # self.prob = 1.0  #[1]
            return int(np.argmax(self.q))

        if self.training:
            epsilon = self.epsilon_sch.update(self.step_in_training).to_float()
        else:
            epsilon = self.config.test_epsilon

        # valid_action_num = self.config.action_space.n - len(invalid_actions)  #[1]
        if random.random() < epsilon:
            action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
            self.q = None
            # self.prob = epsilon / valid_action_num  #[1]
        else:
            self.q = self.parameter.pred_q(state[np.newaxis, ...])[0]
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
                "state": worker.next_state,
                "action": worker.get_onehot_action(),
                "reward": reward,
                "terminated": int(worker.terminated),
                "next_invalid_actions": worker.next_invalid_actions,
                # "prob": self.prob,  # [1]
            }
        )

        self._add_batch(worker)

        if worker.done:
            # 残りstepも追加
            for _ in range(self.config.multisteps - 1):
                worker.add_tracking(
                    {
                        "state": worker.next_state,
                        "action": worker.get_onehot_action(random.randint(0, self.config.action_space.n - 1)),
                        "reward": 0,
                        "terminated": 1,
                        "next_invalid_actions": [],
                        # "prob": 1.0,  # [1]
                    },
                )
                self._add_batch(worker)

    def _add_batch(self, worker: WorkerRun):
        if worker.get_tracking_length() < self.config.multisteps + 1:
            return

        batch = worker.get_trackings(
            [
                "state",
                "action",
                "reward",
                "terminated",
                "next_invalid_actions",
                # "prob"  # [1]
            ],
            size=self.config.multisteps + 1,
        )

        if not self.distributed:
            priority = None
        elif not self.config.memory.requires_priority():
            priority = None
        else:
            if self.q is None:
                self.q = self.parameter.pred_q(worker.state[np.newaxis, ...])[0]
            select_q = self.q[worker.action]
            target_q, _, _ = self.parameter.calc_target_q([batch])
            priority = abs(target_q[0] - select_q)

        self.memory.add(batch, priority)

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.pred_q(worker.state[np.newaxis, ...])[0]
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        worker.print_discrete_action_info(int(maxa), _render_sub)
