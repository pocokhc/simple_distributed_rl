import random
from typing import List, Tuple

import numpy as np

from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.rl.functions.common import inverse_rescaling, render_discrete_action, rescaling

from .rainbow import CommonInterfaceParameter, Config, RemoteMemory


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
def calc_target_q(self: CommonInterfaceParameter, batchs, training: bool):
    batch_size = len(batchs)

    states, n_states, onehot_actions, rewards, dones, _ = zip(*batchs)
    states = np.asarray(states)
    n_states = np.asarray(n_states)
    onehot_actions = np.asarray(onehot_actions)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones)
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
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: CommonInterfaceParameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        assert self.config.multisteps == 1

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

        if self.config.enable_noisy_dense:
            self.q = self.parameter.predict_q(state[np.newaxis, ...])[0]
            self.q[invalid_actions] = -np.inf
            self.action = int(np.argmax(self.q))
            return self.action, {}

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
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
            self.q = None
        else:
            self.q = self.parameter.predict_q(state[np.newaxis, ...])[0]
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            self.action = int(np.argmax(self.q))

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
            self.state,
            next_state,
            np.identity(self.config.action_num, dtype=int)[self.action],
            reward,
            int(not done),
            next_invalid_actions,
        ]

        if not self.distributed:
            td_error = None
        elif not self.config.memory.requires_priority():
            td_error = None
        else:
            if self.q is None:
                self.q = self.parameter.predict_q(self.state[np.newaxis, ...])[0]
            select_q = self.q[self.action]
            target_q = calc_target_q(self.parameter, [batch], training=False)[0]
            td_error = target_q - select_q

        self.remote_memory.add(batch, td_error)
        self.remote_memory.on_step(reward, done)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.predict_q(self.state[np.newaxis, ...])[0]
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
