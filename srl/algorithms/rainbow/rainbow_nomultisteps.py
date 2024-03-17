import random
from typing import Tuple

import numpy as np

from srl.base.define import InfoType
from srl.base.rl.base import RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import inverse_rescaling, render_discrete_action, rescaling
from srl.rl.schedulers.scheduler import SchedulerConfig

from .rainbow import CommonInterfaceParameter, Config


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
def calc_target_q(self: CommonInterfaceParameter, batchs, training: bool):
    batch_size = len(batchs)

    states, n_states, onehot_actions, rewards, dones, _ = zip(*batchs)
    states = np.asarray(states)
    n_states = np.asarray(n_states)
    onehot_actions = np.asarray(onehot_actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    next_invalid_actions = [e for b in batchs for e in b[5]]
    next_invalid_actions_idx = [i for i, b in enumerate(batchs) for e in b[5]]

    n_q_target = self.predict_target_q(n_states)

    # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
    if self.config.enable_double_dqn:
        n_q = self.predict_q(n_states)
        n_q[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q)
        n_act_idx = np.argmax(n_q, axis=1)
        maxq = n_q_target[np.arange(batch_size), n_act_idx]
    else:
        n_q_target[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q_target)
        maxq = np.max(n_q_target, axis=1)

    if self.config.enable_rescale:
        maxq = inverse_rescaling(maxq)

    # --- Q値を計算
    target_q = rewards + dones * self.config.discount * maxq

    if self.config.enable_rescale:
        target_q = rescaling(target_q)

    if training:
        return target_q, states, onehot_actions
    else:
        return target_q


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: CommonInterfaceParameter = self.parameter

        assert self.config.multisteps == 1

        self.step_epsilon = 0

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

    def on_reset(self, worker: WorkerRun) -> InfoType:
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[int, InfoType]:
        state = worker.state
        invalid_actions = worker.get_invalid_actions()

        if self.config.enable_noisy_dense:
            state = self.parameter.create_batch_data(state)
            self.q = self.parameter.predict_q(state)[0]
            self.q[invalid_actions] = -np.inf
            self.action = int(np.argmax(self.q))
            return self.action, {}

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
            self.q = None
        else:
            state = self.parameter.create_batch_data(state)
            self.q = self.parameter.predict_q(state)[0]
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            self.action = int(np.argmax(self.q))

        return self.action, {"epsilon": epsilon}

    def on_step(self, worker: WorkerRun) -> InfoType:
        if not self.training:
            return {}
        reward = worker.reward
        self.step_epsilon += 1

        # reward clip
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
        batch = [
            worker.prev_state,
            worker.state,
            np.identity(self.config.action_num, dtype=int)[self.action],
            reward,
            int(not worker.terminated),
            worker.get_invalid_actions(),
        ]

        if not self.distributed:
            priority = None
        elif not self.config.memory.requires_priority():
            priority = None
        else:
            if self.q is None:
                self.q = self.parameter.predict_q(worker.prev_state[np.newaxis, ...])[0]
            select_q = self.q[self.action]
            target_q = calc_target_q(self.parameter, [batch], training=False)[0]
            priority = abs(target_q - select_q)

        self.memory.add(batch, priority)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.predict_q(worker.prev_state[np.newaxis, ...])[0]
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
