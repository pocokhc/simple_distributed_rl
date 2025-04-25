import random

import numpy as np

from srl.rl.functions import inverse_rescaling, rescaling

from .rainbow import CommonInterfaceParameter, Config, Memory, RLWorker


def calc_target_q(self: CommonInterfaceParameter, batches, training: bool):
    batch_size = len(batches)

    states, n_states, onehot_actions, rewards, undones, _ = zip(*batches)
    states = np.asarray(states)
    n_states = np.asarray(n_states)
    onehot_actions = np.asarray(onehot_actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    undones = np.array(undones, dtype=np.float32)
    next_invalid_actions = [e for b in batches for e in b[5]]
    next_invalid_actions_idx = [i for i, b in enumerate(batches) for e in b[5]]

    n_q_target = self.pred_batch_target_q(n_states)

    # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
    if self.config.enable_double_dqn:
        n_q = self.pred_batch_q(n_states)
        n_q[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q)
        n_act_idx = np.argmax(n_q, axis=1)
        maxq = n_q_target[np.arange(batch_size), n_act_idx]
    else:
        n_q_target[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q_target)
        maxq = np.max(n_q_target, axis=1)

    if self.config.enable_rescale:
        maxq = inverse_rescaling(maxq)

    # --- Q値を計算
    target_q = rewards + undones * self.config.discount * maxq

    if self.config.enable_rescale:
        target_q = rescaling(target_q)

    if training:
        return target_q, states, onehot_actions
    else:
        return target_q


class Worker(RLWorker[Config, CommonInterfaceParameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.step_epsilon = 0
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)

    def policy(self, worker) -> int:
        if self.config.enable_noisy_dense:
            self.q = self.parameter.pred_single_q(worker.state)
            self.q[worker.invalid_actions] = -np.inf
            return int(np.argmax(self.q))

        if self.training:
            epsilon = self.epsilon_sch.update(self.step_in_training).to_float()
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            action = self.sample_action()
            self.q = None
        else:
            self.q = self.parameter.pred_single_q(worker.state)
            self.q[worker.invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            action = int(np.argmax(self.q))

        self.info["epsilon"] = epsilon
        return action

    def on_step(self, worker):
        if not self.training:
            return
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
            worker.state,
            worker.next_state,
            worker.get_onehot_action(),
            reward,
            int(not worker.terminated),
            worker.next_invalid_actions,
        ]

        if not self.distributed:
            priority = None
        elif not self.config.memory.requires_priority():
            priority = None
        else:
            if self.q is None:
                self.q = self.parameter.pred_single_q(worker.state)
            select_q = self.q[worker.action]
            target_q = calc_target_q(self.parameter, [batch], training=False)[0]
            priority = abs(target_q - select_q)

        self.memory.add(batch, priority)

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

        worker.print_discrete_action_info(int(maxa), _render_sub)
